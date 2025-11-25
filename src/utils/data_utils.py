"""
Работа с данными: нормализация, денормализация и подготовка тензоров.
"""
from collections import defaultdict

import numpy as np

from music21 import converter, instrument, note, chord

from fractions import Fraction

from .constants import NUM_PITCHES, MIN_PITCH


def midi_to_notes(midi):
    notes = []
    prev_offset = 0.0
    for n in midi.flat.notes:
        if isinstance(n, note.Note):
            pitch = n.pitch.midi
        elif isinstance(n, chord.Chord):
            pitch = n.pitches[0].midi
        else:
            continue
        step = float(n.offset - prev_offset)
        dur = float(n.quarterLength)
        notes.append({'pitch': pitch, 'step': step, 'duration': dur})
        prev_offset = n.offset
    return notes


def normalize_sequence(seq, pitch_range=(21, 108), step_max=4.0, dur_max=4.0):
    seq = np.array(seq, dtype=np.float32)
    pitches = (seq[..., 0] - pitch_range[0]) / (pitch_range[1] - pitch_range[0])
    steps = seq[..., 1] / step_max
    durations = seq[..., 2] / dur_max
    return np.stack([pitches, steps, durations], axis=-1)


def denormalize_sequence(seq, pitch_range=(21, 108), step_max=4.0, dur_max=4.0):
    seq = np.array(seq)
    pitches = seq[..., 0] * (pitch_range[1] - pitch_range[0]) + pitch_range[0]
    steps = seq[..., 1] * step_max
    durations = seq[..., 2] * dur_max
    return np.stack([pitches, steps, durations], axis=-1)


def prepare_decoder_inputs(targets):
    """
    Формирует dec_inputs и dec_targets для обучения трансформера.
    """
    dec_inputs = np.zeros_like(targets)
    dec_inputs[:, 1:, :] = targets[:, :-1, :]
    return dec_inputs, targets


def get_notes_multitrack(scores):
    notes = []
    for score in scores:
        try:
            # score = converter.parse(path)
            parts = instrument.partitionByInstrument(score)
            if parts:  # несколько партий
                for p in parts.parts:
                    for n in p.recurse().notes:
                        if isinstance(n, note.Note):
                            notes.append({
                                'instrument': p.partName or 'Unknown',
                                'pitch': n.pitch.midi,
                                'step': Fraction(n.offset),
                                'duration': Fraction(n.quarterLength)
                            })
                        elif isinstance(n, chord.Chord):
                            notes.append({
                                'instrument': p.partName or 'Unknown',
                                'pitch': int(np.mean([x.midi for x in n.pitches])),
                                'step': Fraction(n.offset),
                                'duration': Fraction(n.quarterLength)
                            })
        except Exception as e:
            print(f"Ошибка в {score}: {e}")
    return notes


def split_melody_bass_chords_from_notes(notes, melody_threshold=60, chord_threshold=48):
    """
    Простейшее разделение по высоте:
    - >melody_threshold → мелодия
    - между chord_threshold и melody_threshold → аккорды
    - <chord_threshold → бас
    """
    melody, bass, chords = [], [], []
    for n in notes:
        p = n['pitch']
        if p > melody_threshold:
            melody.append(n)
        elif p < chord_threshold:
            bass.append(n)
        else:
            chords.append(n)
    return melody, bass, chords


def normalize_sequence_global(seq, stats):
    """
    seq: (N,3) raw: pitch(int), step, duration
    Возвращает (N,3) где:
      - column0 = PITCH (оставляем как integer value — но как float dtype)
      - column1 = step normalized
      - column2 = duration normalized
    """
    seq = np.array(seq, dtype=np.float32)
    pitch = seq[..., 0]  # сохраняем integer (как float)
    step = seq[..., 1] / stats['step_max']
    dur = seq[..., 2] / stats['dur_max']
    return np.stack([pitch, step, dur], axis=-1)


def denormalize_sequence_global(seq_norm, stats):
    seq = np.array(seq_norm, dtype=np.float32)
    pitch = seq[..., 0] * (stats['pitch_max'] - stats['pitch_min']) + stats['pitch_min']
    step = seq[..., 1] * stats['step_max']
    dur = seq[..., 2] * stats['dur_max']
    return np.stack([pitch, step, dur], axis=-1)


def denormalize_sequence_global(seq_norm, stats):
    """
    seq_norm shape (T,3) or (batch, T, 3) — ожидает pitch in integer values in col 0 already,
    returns pitch, step, dur denormalized
    """
    a = np.array(seq_norm, dtype=np.float32)
    if a.ndim == 3:
        pitch = a[..., 0]
        step = a[..., 1] * stats['step_max']
        dur = a[..., 2] * stats['dur_max']
        return np.stack([pitch, step, dur], axis=-1)
    else:
        pitch = a[:, 0]
        step = a[:, 1] * stats['step_max']
        dur = a[:, 2] * stats['dur_max']
        return np.stack([pitch, step, dur], axis=-1)


def pad_sequence(seq, target_len):
    """
    seq: np.array (N, L, F)
    target_len: int, желаемая длина по time axis (L)
    """
    N, L, F = seq.shape
    if L >= target_len:
        return seq[:, :target_len, :]
    pad_len = target_len - L
    pad = np.zeros((N, pad_len, F), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=1)


def split_parts_basic(notes, melody_thr=64, bass_thr=52):
    melody, chords, bass = [], [], []
    for n in notes:
        if n['pitch'] > melody_thr:
            melody.append(n)
        elif n['pitch'] < bass_thr:
            bass.append(n)
        else:
            chords.append(n)
    return melody, chords, bass


def notes_with_offsets_from_midi(notes):
    # если у тебя notes уже содержит 'offset' — используй его.
    # В твоём midi_to_notes offset не сохранялся, но мы можем восстановить:
    offs = []
    prev = 0.0
    for n in notes:
        # здесь предполагаем что notes идут в порядке offset (как у midi.flat.notes)
        # и что step соответствует delta offset
        prev += float(n.get("step", 0.0))
        offs.append(prev)
    # возвращаем новые объекты с offset
    new_notes = []
    for n, o in zip(notes, offs):
        new = dict(n)
        new["offset"] = o
        new_notes.append(new)
    return new_notes


def split_parts_by_onset(notes, window=1e-6):
    """
    notes: list of dict {'pitch','step','duration', 'offset' (opt)}
    window: группировка по offset: считаем offsets равными если |o1-o2| <= window
            (window в тех же единицах offset; для quarterLength-осн. музыки можно взять 1e-3..1e-1)
    Возвращает (melody, chords, bass) — списки note dicts.
    """
    # убедимся, что notes содержат offset
    if "offset" not in notes[0]:
        notes = notes_with_offsets_from_midi(notes)

    # сгруппируем по округлённому offset (или кластерим вручную)
    # для устойчивости округлим offset к квантам:
    # подобрать квант (quant) в зависимости от step scale; можно использовать window.
    quant = window
    groups = defaultdict(list)
    for n in notes:
        o = float(n["offset"])
        key = round(o / quant)  # integer bucket
        groups[key].append(n)

    melody, chords, bass = [], [], []
    # сортируем группы по времени
    for key in sorted(groups.keys()):
        grp = groups[key]
        # если в группе только одна нота — это мелодия (или chords), решаем так:
        if len(grp) == 1:
            n = grp[0]
            melody.append(n)  # можно решение: treat singletons as melody
            continue

        # иначе найдём min/max pitch
        pitches = [x["pitch"] for x in grp]
        max_idx = int(np.argmax(pitches))
        min_idx = int(np.argmin(pitches))
        # append highest to melody, lowest to bass, rest to chords
        melody.append(grp[max_idx])
        bass.append(grp[min_idx])
        for i, x in enumerate(grp):
            if i == max_idx or i == min_idx:
                continue
            chords.append(x)

    return melody, chords, bass


def make_decoder_inputs_from_targets(target_seq):
    """
    target_seq: (N, seq_len, 3)
    return dec_input: shift-right with zeros at first timestep
    """
    N, L, F = target_seq.shape
    dec_in = np.zeros_like(target_seq)
    dec_in[:, 1:, :] = target_seq[:, :-1, :]
    return dec_in


def pad_sequence(seq, target_len):
    pad_len = target_len - seq.shape[0]
    if pad_len <= 0:
        return seq[:target_len]
    else:
        pad = np.zeros((pad_len, seq.shape[1]), dtype=seq.dtype)
        return np.vstack([seq, pad])


def pad_sequence_3d(seq, target_len):
    """
    seq: np.array (N, L, F)
    target_len: int, желаемая длина по оси L
    """
    N, L, F = seq.shape
    if L >= target_len:
        return seq[:, :target_len, :]
    pad_len = target_len - L
    pad = np.zeros((N, pad_len, F), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=1)


def prepare_sequences_(notes, stats, rare_note, seq_length=50):
    if len(notes) < seq_length + 1:
        return None, None
    raw = np.array([[n["pitch"], n["step"], n["duration"]] for n in notes if n["pitch"] not in rare_note],
                   dtype=np.float32)
    norm = normalize_sequence_global(raw, stats)
    X, y = [], []
    for i in range(len(norm) - seq_length):
        X.append(norm[i:i + seq_length])
        y.append(norm[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_sequences(notes, stats, rare_note=None, seq_length=50):
    """
    notes: список словарей с 'pitch', 'step', 'duration'
    возвращает:
      X: (N, seq_len, 3)
      y_pitch: (N,1 int)
      y_step:  (N,1 float)
      y_dur:   (N,1 float)
    """
    if len(notes) < seq_length + 1:
        return None, None, None, None

    raw = np.array([[n["pitch"], n["step"], n["duration"]] for n in notes if n["pitch"] not in rare_note],
                   dtype=np.float32)
    norm = normalize_sequence_global(raw, stats)

    X, y_pitch, y_step, y_dur = [], [], [], []
    pitches_int = raw[:, 0].astype(int)
    steps = raw[:, 1]
    durs = raw[:, 2]

    for i in range(len(norm) - seq_length):
        X.append(norm[i:i + seq_length])
        y_pitch.append(pitches_int[i + seq_length])
        y_step.append(steps[i + seq_length])
        y_dur.append(durs[i + seq_length])

    return np.array(X, dtype=np.float32), \
        np.array(y_pitch, dtype=np.int32)[:, None], \
        np.array(y_step, dtype=np.float32)[:, None], \
        np.array(y_dur, dtype=np.float32)[:, None]


def process_midi_for_training(midi_path, stats, pitch_offset, seq_len=50, L_enc=100):
    """
    Возвращает:
        enc_in       -> (N, L_enc, 3)
        dec_in       -> (N, seq_len, 3)
        pitch_targets -> (N, seq_len) индексы классов [0, NUM_PITCHES-1]
        step_targets  -> (N, seq_len, 1)
        dur_targets   -> (N, seq_len, 1)
    """

    notes = midi_to_notes(midi_path)
    melody, chords, bass = split_parts_basic(notes)

    # Преобразуем последовательности
    mX, mY_pitch, mY_step, mY_dur = prepare_sequences(melody, stats, seq_len)
    bX, bY_pitch, bY_step, bB_dur = prepare_sequences(bass, stats, seq_len)
    cX, cY_pitch, cY_step, cY_dur = prepare_sequences(chords, stats, seq_len)

    if mX is None or bX is None or cX is None:
        return None

    # Усечь до минимальной длины по сэмплам
    L_samples = min(len(mX), len(bX), len(cX))
    mX, bX, cX = mX[:L_samples], bX[:L_samples], cX[:L_samples]
    cY_pitch, cY_step, cY_dur = cY_pitch[:L_samples], cY_step[:L_samples], cY_dur[:L_samples]

    # encoder input: concat по времени (axis=1)
    enc_in = np.concatenate([mX, bX], axis=1)
    enc_in = pad_sequence_3d(enc_in, L_enc)

    # decoder input: shift-right
    dec_in = make_decoder_inputs_from_targets(cX)

    N, L_dec, F = cX.shape

    # --- Pitch targets: индексы (не one-hot), для всех timestep'ов ---
    pitch_targets = np.zeros((N, L_dec), dtype=np.int32)
    for i in range(N):
        # проверяем, что ноты не выходят за границы
        safe_pitch = np.clip(cY_pitch[i] - pitch_offset, 0, NUM_PITCHES - 1)
        pitch_targets[i] = safe_pitch

    # step/duration targets: shape (N, L_dec, 1)
    step_targets = np.expand_dims(cY_step, axis=-1)
    dur_targets = np.expand_dims(cY_dur, axis=-1)

    return enc_in, dec_in, pitch_targets, step_targets, dur_targets
