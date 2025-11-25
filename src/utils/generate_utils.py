"""
Генерация последовательностей и конвертация в music21 объекты.
"""
import numpy as np

import tensorflow as tf

from music21 import note, chord, instrument, stream

from .data_utils import denormalize_sequence_global

from .constants import MIN_PITCH


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


def generate_full_autoregressive(
        model, enc_input, seed_dec,
        length=64,
        pitch_temp=1.0,
        cont_temp=0.0,
        # если 0 -> deterministic (take mean / model output), >0 -> add gaussian noise with std=cont_temp*scale
        cont_scale=None,  # e.g. {'step': stats['step_max'], 'dur': stats['dur_max'] }
        sampling_mode='stochastic',  # 'stochastic' or 'argmax'
        stats=None
):
    """
    Возвращает sequence (1, length+1, 3) — первый элемент seed_dec.
    - cont_temp: если >0, добавляем шум N(0, cont_temp * scale)
    - sampling_mode: 'stochastic' -> sample from softmax; 'argmax' -> take argmax
    """
    generated = [seed_dec.copy()]

    prev_step = 0.0

    # если cont_scale не задан — берем из stats
    if cont_scale is None and stats is not None:
        cont_scale = {'step': float(stats.get('step_max', 1.0)), 'dur': float(stats.get('dur_max', 1.0))}
    if cont_scale is None:
        cont_scale = {'step': 1.0, 'dur': 1.0}

    for t in range(length):
        dec_input = np.concatenate(generated, axis=1)  # (1, t+1, 3)
        pitch_logits, step_pred, dur_pred = model.predict([enc_input, dec_input], verbose=0)
        # --- PITCH: logits -> probs ---
        logits = pitch_logits[:, -1, :]  # (1, num_classes)
        if sampling_mode == 'argmax' or pitch_temp == 0.0:
            pitch_next_idx = np.argmax(logits, axis=-1)  # (1,)
        else:
            probs = tf.nn.softmax(logits / float(pitch_temp)).numpy()
            probs = np.nan_to_num(probs, nan=1e-8)
            probs /= np.sum(probs, axis=-1, keepdims=True)
            pitch_next_idx = np.array([np.random.choice(probs.shape[-1], p=probs[0])], dtype=np.int32)

        pitch_next = pitch_next_idx.reshape(1, 1, 1).astype(np.float32)

        # --- STEP / DURATION ---
        step_val = step_pred[:, -1, 0].reshape(1, 1, 1).astype(
            np.float32)  # model already outputs non-negative if relu used
        dur_val = dur_pred[:, -1, 0].reshape(1, 1, 1).astype(np.float32)

        # if NaN -> fallback to small value
        if np.isnan(step_val).any(): step_val[:] = 0.25
        if np.isnan(dur_val).any():  dur_val[:] = 0.25

        # stochastic cont sampling: add gaussian noise scaled by cont_temp*cont_scale
        if sampling_mode == 'stochastic' and cont_temp > 0.0:
            step_std = cont_temp * cont_scale['step']
            dur_std = cont_temp * cont_scale['dur']
            step_val = step_val + np.random.normal(scale=step_std, size=step_val.shape).astype(np.float32)
            dur_val = dur_val + np.random.normal(scale=dur_std, size=dur_val.shape).astype(np.float32)

        # enforce non-negative (safety)
        step_val = np.maximum(step_val, 0.0)
        dur_val = np.maximum(dur_val, 0.0)

        step_val += prev_step
        prev_step += step_val
        next_token = np.concatenate([pitch_next, step_val, dur_val], axis=-1)
        generated.append(next_token)

    return np.concatenate(generated, axis=1)  # (1, length+1, 3)


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
            n = chord.Chord([p, p + 4, p + 7])
        else:
            n = note.Note(p)
        n.quarterLength = dur
        part.insert(offset, n)
        offset += step
    return part


def sample_from_probs(probs, temperature=1.0, top_k=None, top_p=None):
    """
    probs: 1D array of class probabilities
    returns index
    """
    p = np.array(probs, dtype=np.float64)
    # temperature
    if temperature != 1.0:
        p = np.log(p + 1e-9) / temperature
        p = np.exp(p)
    p = p / np.sum(p)
    if top_k is not None:
        # zero out everything except top_k
        ind = np.argsort(p)[-top_k:]
        mask = np.zeros_like(p, dtype=bool)
        mask[ind] = True
        p = p * mask
        p = p / np.sum(p)
    return np.random.choice(len(p), p=p)


def generate_autoregressive_sample(model, enc_input, seed_dec=None, length=128,
                                   stats=None, temperature=0.9, res_std=(0.02, 0.02)):
    """
    enc_input: (1, L_enc, 6)
    seed_dec: (1, 1, 3) or None -> zeros
    returns denormalized array (T,3) with pitch as integer MIDI, step,dur denormalized
    """
    if seed_dec is None:
        seed_dec = np.zeros((1, 1, 3), dtype=np.float32)

    dec_seq = seed_dec.copy()  # (1, cur_T, 3)

    # estimate residual std if not provided — here passed or default
    for _ in range(length - dec_seq.shape[1]):
        p_pred, s_pred, d_pred = model.predict([enc_input, dec_seq], verbose=0)
        # p_pred: (1, T, NUM_PITCHES)
        # take last time step probabilities
        probs = p_pred[0, -1, :]
        pitch_idx = sample_from_probs(probs, temperature=temperature, top_k=12)
        pitch_midi = int(pitch_idx + MIN_PITCH)

        # step/duration are regression outputs: take last predicted value and add gaussian noise scaled by res_std
        step_val = float(s_pred[0, -1, 0]) + np.random.normal(scale=res_std[0])
        dur_val = float(d_pred[0, -1, 0]) + np.random.normal(scale=res_std[1])

        # clip normalized step/dur to [0,1]
        step_val = float(np.clip(step_val, 0.0, 1.0))
        dur_val = float(np.clip(dur_val, 0.0, 1.0))

        next_token = np.array([[[pitch_midi, step_val, dur_val]]], dtype=np.float32)  # (1,1,3)
        dec_seq = np.concatenate([dec_seq, next_token], axis=1)

    # dec_seq shape (1, T, 3) where col0 = pitch MIDI ints; denormalize step/dur
    gen = dec_seq[0]  # (T,3)
    # denormalize step/dur and keep pitch as int
    den = denormalize_sequence_global(gen, stats)  # pitch preserved, step/dur denorm
    # round pitch to int
    den[:, 0] = np.round(den[:, 0]).astype(np.int32)
    return den  # shape (T,3)
