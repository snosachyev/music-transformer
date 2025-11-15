"""
Генерация последовательностей и конвертация в music21 объекты.
"""
import numpy as np
from music21 import note, chord, instrument, stream


def generate_autoregressive(model, enc_input, seed_dec, length=32):
    """
    Авторегрессивная генерация по шагам.
    """
    dec_seq = seed_dec.copy()
    for _ in range(length - seed_dec.shape[1]):
        preds = model.predict([enc_input, dec_seq], verbose=0)
        next_step = preds[:, -1:, :]
        dec_seq = np.concatenate([dec_seq, next_step], axis=1)
    return dec_seq


def generate_full_autoregressive(model, enc_input, seed_dec, length=32):
    dec_seq = seed_dec.copy()
    for _ in range(length - seed_dec.shape[1]):
        preds = model.predict([enc_input, dec_seq], verbose=0)
        pitch_pred, step_pred, dur_pred = preds
        next_pitch = pitch_pred[:, -1:, :]
        next_step = step_pred[:, -1:, :]
        next_dur = dur_pred[:, -1:, :]
        next_token = np.concatenate([next_pitch, next_step, next_dur], axis=-1)
        dec_seq = np.concatenate([dec_seq, next_token], axis=1)
    return dec_seq


def generated_to_part(generated, name="Generated", make_chords=False):
    """
    Превращает предсказанную последовательность в music21.Part
    """
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.partName = name
    offset = 0.0
    for p, step, dur in generated[0]:
        p, step, dur = int(round(p)), float(step), float(dur)
        step, dur = max(0.25, step), max(0.25, dur)
        if make_chords:
            n = chord.Chord([p, p+4, p+7])
        else:
            n = note.Note(p)
        n.quarterLength = dur
        part.insert(offset, n)
        offset += step
    return part
