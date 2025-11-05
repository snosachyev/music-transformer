"""
Работа с данными: нормализация, денормализация и подготовка тензоров.
"""
import numpy as np

from music21 import converter, instrument, note, chord

from fractions import Fraction


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
            #score = converter.parse(path)
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


def prepare_sequences(notes, seq_length=32,
                      pitch_range=(21, 108),  # диапазон MIDI нот
                      step_max=4.0,           # макс. шаг (в четвертях)
                      dur_max=4.0,            # макс. длительность
                      normalize=True):
    """
    notes: список словарей или списков вида [pitch, step, duration]
    seq_length: длина окна последовательности
    normalize: если True — нормализует признаки в диапазон [0,1]
    """

    if len(notes) < seq_length:
        return None, None

    # --- Поддержка обоих форматов данных ---
    if isinstance(notes[0], (list, tuple)):
        data = np.array(notes, dtype=float)
        pitches, steps, durations = data[:, 0], data[:, 1], data[:, 2]
    else:
        pitches = np.array([n["pitch"] for n in notes], dtype=float)
        steps = np.array([n["step"] for n in notes], dtype=float)
        durations = np.array([n["duration"] for n in notes], dtype=float)

    # --- Нормализация ---
    if normalize:
        pitches = (pitches - pitch_range[0]) / (pitch_range[1] - pitch_range[0])
        steps = np.clip(steps / step_max, 0, 1)
        durations = np.clip(durations / dur_max, 0, 1)

    # --- Формирование обучающих последовательностей ---
    X, y = [], []
    for i in range(len(pitches) - seq_length):
        X.append(np.stack([pitches[i:i+seq_length],
                           steps[i:i+seq_length],
                           durations[i:i+seq_length]], axis=1))
        y.append(np.stack([steps[i+1:i+seq_length+1],
                           pitches[i+1:i+seq_length+1],
                           durations[i+1:i+seq_length+1]], axis=1))

    return np.array(X, dtype=float), np.array(y, dtype=float)
