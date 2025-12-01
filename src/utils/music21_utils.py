"""
Парсинг MIDI → ноты, разделение партий, создание Score.
"""

import os

import numpy as np

from typing import List, Dict, Any

from music21 import converter, instrument, note, chord, stream, volume, midi, clef


def compute_global_stats(all_notes):
    arr = np.array([[n['pitch'], n['step'], n['duration']] for n in all_notes], dtype=np.float32)
    pitch_min = np.min(arr[:, 0])
    pitch_max = np.max(arr[:, 0])
    step_max = float(np.percentile(arr[:, 1], 99.5))
    dur_max = float(np.percentile(arr[:, 2], 99.5))
    return {'pitch_min': float(pitch_min),
            'pitch_max': float(pitch_max),
            'step_max': max(step_max, 1e-6),
            'dur_max': max(dur_max, 1e-6)
            }


def get_all_midis(path, midis=None):
    if midis is None:
        midis = []

    for name in os.listdir(path):
        full_path = os.path.join(path, name)

        if os.path.isdir(full_path):
            # рекурсивный обход
            get_all_midis(full_path, midis)

        elif name.lower().endswith(".mid"):
            try:
                midi = converter.parse(full_path)
                midis.append(midi)
            except Exception as e:
                print(f"Ошибка при чтении {full_path}: {e}")

    return midis


# -------------------------
# ONSET / TIMELINE helpers
# -------------------------
def seq_to_onsets(seq):
    """
    seq: numpy array shape (T,3) where seq[:,0]=pitch, seq[:,1]=step, seq[:,2]=dur
    returns:
      onsets: numpy (T,) absolute onsets (first onset typically 0)
      durations: numpy (T,) durations
      pitches: numpy (T,) pitch values
    """
    seq = np.asarray(seq, dtype=float)
    steps = seq[:, 1].astype(float)
    durs = seq[:, 2].astype(float)
    onsets = np.cumsum(steps) - steps[0]  # if first step is 0 -> first onset 0
    pitches = seq[:, 0].astype(int)
    return onsets, durs, pitches


def onsets_to_seq(onsets, durations, pitches):
    """
    reverse: build seq (T,3) from onsets/durations/pitches. step = delta onsets (first step = 0)
    """
    onsets = np.asarray(onsets, dtype=float)
    durations = np.asarray(durations, dtype=float)
    pitches = np.asarray(pitches, dtype=float)
    steps = np.concatenate(([0.0], np.diff(onsets)))
    seq = np.stack([pitches, steps, durations], axis=-1)
    return seq


def align_timelines(enc1_seq, enc2_seq, gen_seq, method='clip'):
    """
    Выравнивает временные границы трёх партий.
    Все seq в формате numpy (T,3) с реальными step/dur (уже денормализованными).
    method:
      - 'clip'  : если gen длинее, отрезать лишние ноты, если короче — оставить;
      - 'scale' : масштабировать onsets генерируемой партии, чтобы длина совпала с эталоном.
    Возвращает новые seq: enc1_seq, enc2_seq, gen_seq_aligned (в формате (T,3) с recomputed steps).
    """
    # переводим в onsets
    on1, d1, p1 = seq_to_onsets(enc1_seq)
    on2, d2, p2 = seq_to_onsets(enc2_seq) if len(enc2_seq) > 0 else (np.array([]), np.array([]), np.array([]))
    on_g, d_g, p_g = seq_to_onsets(gen_seq)

    # эталонная длина — обычно берем max(len(enc1), len(enc2))
    ref_len = 0.0
    if on1.size:
        ref_len = max(ref_len, on1[-1] + d1[-1])
    if on2.size:
        ref_len = max(ref_len, on2[-1] + d2[-1])
    # если оба энкодера пусты, берем ген как эталон
    if ref_len == 0.0 and on_g.size:
        ref_len = on_g[-1] + d_g[-1]

    if ref_len == 0.0:
        # все пустые — ничего делать
        return enc1_seq, enc2_seq, gen_seq

    gen_end = on_g[-1] + d_g[-1]

    if method == 'clip':
        # отрезаем ноты генерируемой партии, которые начинаются за пределом ref_len
        keep_mask = (on_g < ref_len)
        on_g_new = on_g[keep_mask]
        d_g_new = d_g[keep_mask]
        p_g_new = p_g[keep_mask]
        # recompute steps
        gen_seq_new = onsets_to_seq(on_g_new, d_g_new, p_g_new) if on_g_new.size else np.zeros((0, 3))
    elif method == 'scale' and gen_end > 0:
        scale = ref_len / gen_end
        on_g_scaled = on_g * scale
        d_g_scaled = d_g * scale
        gen_seq_new = onsets_to_seq(on_g_scaled, d_g_scaled, p_g)
    else:
        gen_seq_new = gen_seq

    # Для энкодерных партий steps могут оставаться без изменений — но можно также подрезать/расширить
    return enc1_seq, enc2_seq, gen_seq_new


# -------------------------
# MIDI saving (music21)
# -------------------------
def seq_to_part(seq, part_name="Part", instr_name="Piano"):
    """
    seq: numpy (T,3) where seq[:,0]=pitch midi int, seq[:,1]=step delta, seq[:,2]=dur (quarterLength)
    Возвращает music21.stream.Part с установленными offset'ами.
    """
    part = stream.Part()
    # set instrument
    try:
        instr = instrument.fromString(instr_name)
        part.insert(0, instr)
    except Exception:
        pass

    if seq is None or len(seq) == 0:
        return part

    seq = np.asarray(seq, dtype=float)
    steps = seq[:, 1]
    durs = seq[:, 2]
    pitches = seq[:, 0].astype(int)

    # compute offsets
    offsets = np.cumsum(steps) - steps[0]
    # if first step not zero, normalize to 0
    if offsets.size and offsets[0] != 0:
        offsets = offsets - offsets[0]

    for p, off, dur in zip(pitches, offsets, durs):
        if np.isnan(p) or np.isinf(p):
            continue
        try:
            n = note.Note(int(np.clip(p, 0, 127)))
            n.quarterLength = max(0.0625, float(dur))  # защита от нулей/отриц.
            part.insert(float(off), n)
        except Exception:
            # если не удалось (например, pitch 0 как rest?), попробуем rest
            r = note.Rest()
            r.quarterLength = max(0.0625, float(dur))
            part.insert(float(off), r)
    return part


def save_multi_track_midi(enc1_seq, enc2_seq, gen_seq, fp="out.mid",
                          instr1="Piano", instr2="Piano", instr_gen="Accordion"):
    """
    Собирает score с 2 или 3 партиями и сохраняет в MIDI.
    enc1_seq, enc2_seq, gen_seq — numpy (T,3) или списки кортежей (pitch,step,dur).
    """
    # to numpy
    enc1 = np.asarray(enc1_seq, dtype=float) if len(enc1_seq) else np.zeros((0, 3))
    enc2 = np.asarray(enc2_seq, dtype=float) if len(enc2_seq) else np.zeros((0, 3))
    gen = np.asarray(gen_seq, dtype=float) if len(gen_seq) else np.zeros((0, 3))

    score = stream.Score()
    if enc1.size:
        p1 = seq_to_part(enc1, part_name="Part1", instr_name=instr1)
        score.insert(0, p1)
    if enc2.size:
        p2 = seq_to_part(enc2, part_name="Part2", instr_name=instr2)
        score.insert(0, p2)
    if gen.size:
        pg = seq_to_part(gen, part_name="Generated", instr_name=instr_gen)
        score.insert(0, pg)

    score.write('midi', fp=fp)
    return fp


def detect_instruments_in_midi(score: List) -> List[str]:
    """Return a list of instrument identifiers found in a MIDI file using music21."""

    instruments = set()
    for part in instrument.partitionByInstrument(score):
        inst_name = part.partName
        instruments.add(inst_name)
    return list(instruments)


# -----------------------------
# Extract per-sample dict with empty lists for missing instruments
# -----------------------------
def extract_sample(score, global_instruments: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    sample = {inst: [] for inst in global_instruments}
    instrument_part = instrument.partitionByInstrument(score)
    for part in instrument_part:
        inst_name = part.partName

        notes = []
        for n in part:
            if isinstance(n, note.Note):
                notes.append({
                    'start': float(n.offset),
                    'pitch': n.pitch.midi,
                    'dur': float(n.quarterLength),
                    'velocity': getattr(n, 'volume', volume.Volume()).velocity or 64
                })
            elif isinstance(n, chord.Chord):
                for p in n.pitches:
                    notes.append({
                        'start': float(n.offset),
                        'pitch': p.midi,
                        'dur': float(n.quarterLength),
                        'velocity': getattr(n, 'volume', volume.Volume()).velocity or 64
                    })
        sample[inst_name] = notes
    return sample
