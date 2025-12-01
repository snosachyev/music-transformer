"""
Работа с данными: нормализация, денормализация и подготовка тензоров.
"""
import numpy as np

from music21 import converter, instrument, note, chord


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


# 5) Денормализация/подготовка энкодерных партий (если они были нормализованы)
# Если enc_mel и enc_bass сейчас в нормализованном виде, применяем denormalize
def try_denorm(enc_np, stats):
    if enc_np is None or enc_np.size == 0:
        return np.zeros((0, 3))
    enc_np = np.asarray(enc_np)
    if enc_np.ndim == 3 and enc_np.shape[0] == 1:
        enc_np = enc_np[0]
    # если похоже на нормализованные (в пределах 0..1), денормализуем
    try:
        den = denormalize_sequence_global(enc_np[np.newaxis, ...], stats)
        if den.ndim == 3:
            return den[0]
        return den
    except Exception:
        # fallback — вернуть исходное
        return enc_np
