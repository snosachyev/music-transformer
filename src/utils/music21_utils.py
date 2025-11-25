"""
Парсинг MIDI → ноты, разделение партий, создание Score.
"""

import glob

import os

import numpy as np

from music21 import converter, instrument, note, chord, stream

from .data_utils import (
    split_parts_by_onset, prepare_sequences, pad_sequence_3d,
    make_decoder_inputs_from_targets, midi_to_notes)


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


def split_parts(notes, melody_thr=64, bass_thr=52):
    melody, chords, bass = [], [], []
    for n in notes:
        if n['pitch'] > melody_thr:
            melody.append(n)
        elif n['pitch'] < bass_thr:
            bass.append(n)
        else:
            chords.append(n)
    return melody, chords, bass


def combine_to_score(parts):
    s = stream.Score()
    for p in parts:
        s.insert(0, p)
    return s


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

def get_midis_from_folder(path):
    return glob.glob(path + "/*.mid")


def sequence_to_part_from_seq(seq, instr_name="Piano", pitch_offset=0, denom_step=1.0):
    """
    seq: (T, 3) — pitch, step, dur  (pitch in MIDI numbers or indices already denormed to MIDI)
    instr_name: e.g. "Piano Right Hand"
    pitch_offset: если pitch это индекс (0..), добавь offset к получить MIDI
    denom_step: если шагы — это времена в четвертях (already), иначе подстрой
    """
    p = np.array(seq)
    part = stream.Part()
    part.append(instrument.fromString(instr_name))
    offset = 0.0
    prev_offset = 0.0
    for (pitch, step, dur) in p:
        # protect values
        if np.isnan(pitch) or pitch <= 0:
            # можно вставить rest
            n = note.Rest()
        else:
            midi_pitch = int(np.round(pitch + pitch_offset))
            n = note.Note(midi_pitch)
        # step — delta to shift from prev
        offset = prev_offset + float(max(step, 0.0))
        n.offset = offset
        n.quarterLength = max(0.125, float(max(dur, 0.125)))  # минимальная длительность 1/8
        part.insert(offset, n)
        prev_offset = offset
    return part


from music21 import stream, note, chord, instrument, midi, clef


def save_three_voice_midi(enc1_seq, enc2_seq, gen_seq, fp="three_voice.mid"):
    """
    Сохраняет три партии в один MIDI файл и визуализирует нотный стан.

    Args:
        enc1_seq: np.array (T,3) - мелодия
        enc2_seq: np.array (T,3) - бас
        gen_seq: np.array (T,3) - сгенерированная партия
        fp: путь для сохранения MIDI
    """
    # Создаем stream для каждой партии
    melody_part = stream.Part()
    melody_part.insert(0, instrument.Piano())
    melody_part.insert(0, clef.TrebleClef())

    bass_part = stream.Part()
    bass_part.insert(0, instrument.Piano())
    bass_part.insert(0, clef.BassClef())

    gen_part = stream.Part()
    gen_part.insert(0, instrument.Piano())
    gen_part.insert(0, clef.TrebleClef())  # можно менять в зависимости от партии

    # Функция для добавления нот в партитуру
    def add_notes_to_part(part, seq):
        time_cursor = 0.0
        for pitch_val, step_val, dur_val in seq:
            # Если step отрицательный, зафиксируем 0
            step_val = max(step_val, 0.0)
            time_cursor += step_val
            # Создаем ноту
            n = note.Note(int(round(pitch_val)))
            n.quarterLength = max(dur_val, 0.01)
            part.insert(time_cursor, n)

    add_notes_to_part(melody_part, enc1_seq)
    add_notes_to_part(bass_part, enc2_seq)
    add_notes_to_part(gen_part, gen_seq)

    # Создаем полный stream
    score = stream.Score()
    score.insert(0, melody_part)
    score.insert(0, bass_part)
    score.insert(0, gen_part)

    # Сохраняем MIDI
    score.write('midi', fp)

    # Визуализируем нотный стан (например, для Jupyter Notebook)
    #score.show('musicxml')


def normalize_pitch(pitch_array, pitch_classes):
    pitch = np.nan_to_num(pitch_array, nan=0.0)
    pitch = np.clip(pitch, 0, pitch_classes - 1)
    pitch = pitch.astype(np.int32)
    return pitch


def normalize_continuous(arr, max_allowed):
    arr = np.nan_to_num(arr, nan=0.0)
    arr[arr < 0] = 0.0
    arr[arr > max_allowed] = max_allowed
    return arr


def check_step_dur(step_list, dur_list, global_step_max, global_dur_max):
    print("=== CHECK STEP ===")
    all_steps = np.concatenate(step_list)
    print("min:", np.nanmin(all_steps))
    print("max:", np.nanmax(all_steps))
    print("NaN:", np.isnan(all_steps).sum())
    print("negatives:", (all_steps < 0).sum())
    print("out_of_range:", (all_steps > global_step_max).sum())

    print("\n=== CHECK DURATION ===")
    all_dur = np.concatenate(dur_list)
    print("min:", np.nanmin(all_dur))
    print("max:", np.nanmax(all_dur))
    print("NaN:", np.isnan(all_dur).sum())
    print("negatives:", (all_dur < 0).sum())
    print("out_of_range:", (all_dur > global_dur_max).sum())


def process_midi(midi, stats, pitch_offset,
                 rare_note=None,
                 seq_len=50, L_enc=100,
                 normalization=False):
    notes = midi_to_notes(midi)
    melody, chords, bass = split_parts_by_onset(notes)
    mX, mY_pitch, mY_step, mY_dur = prepare_sequences(melody, stats, rare_note, seq_len)
    bX, bY_pitch, bY_step, bY_dur = prepare_sequences(bass, stats, rare_note, seq_len)
    cX, cY_pitch, cY_step, cY_dur = prepare_sequences(chords, stats, rare_note, seq_len)

    if mX is None or bX is None or cX is None:
        return None

    L_samples = min(len(mX), len(bX), len(cX))
    mX, bX, cX = mX[:L_samples], bX[:L_samples], cX[:L_samples]
    cY_pitch, cY_step, cY_dur = (
        cY_pitch[:L_samples],
        cY_step[:L_samples],
        cY_dur[:L_samples]
    )

    # encoder input
    enc_in = np.concatenate([mX, bX], axis=1)
    enc_in = pad_sequence_3d(enc_in, L_enc)

    # decoder input
    dec_in = make_decoder_inputs_from_targets(cX)

    N, L_dec, F = cX.shape

    # pitch targets (index)
    pitch_targets = np.zeros((N, L_dec), dtype=np.int32)
    for i in range(N):
        pitch_targets[i] = cY_pitch[i] - pitch_offset

    # continuous targets
    step_targets = np.expand_dims(cY_step, axis=-1)
    dur_targets = np.expand_dims(cY_dur, axis=-1)

    # ----- ДОПОЛНИТЕЛЬНАЯ НОРМАЛИЗАЦИЯ -----
    if normalization:
        pitch_classes = int(stats['pitch_max'] - stats['pitch_min'] + 1)

        pitch_targets = normalize_pitch(pitch_targets, pitch_classes)
        step_targets = normalize_continuous(step_targets, stats["step_max"])
        dur_targets = normalize_continuous(dur_targets, stats["dur_max"])

    return enc_in, dec_in, pitch_targets, step_targets, dur_targets


def split_encoder_input(enc_in, seq_len=50):
    """
    enc_in: (N, L_enc, 3)
    Первая половина — мелодия
    Вторая половина — бас
    """
    # мелодия = первые seq_len шагов
    enc1 = enc_in[:, :seq_len, :]

    # бас = следующие seq_len шагов
    enc2 = enc_in[:, seq_len:seq_len * 2, :]

    return enc1, enc2







# -------------------------
# 1) Утилиты
# -------------------------
def safe_softmax(logits, axis=-1, eps=1e-8):
    """Softmax с защитой от NaN/overflow."""
    logits = np.asarray(logits, dtype=np.float32)
    if np.isnan(logits).any():
        logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    ex = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    s = np.sum(ex, axis=axis, keepdims=True)
    s = np.maximum(s, eps)
    return ex / s


def clip_and_fix_step_dur(step, dur, step_min=1e-3, step_max=1.0, dur_min=1e-3, dur_max=4.0, smooth_alpha=0.2,
                          prev_step=None):
    """Клиппинг и небольшое сглаживание step/dur. Возвращает (step_fixed, dur_fixed)."""
    s = float(step)
    d = float(dur)
    if np.isnan(s) or np.isinf(s):
        s = step_min
    if np.isnan(d) or np.isinf(d):
        d = dur_min
    # clip
    s = max(step_min, min(step_max, s))
    d = max(dur_min, min(dur_max, d))
    # smooth relative to prev step (если задан)
    if prev_step is not None:
        s = (1.0 - smooth_alpha) * prev_step + smooth_alpha * s
    return s, d


# -------------------------
# 2) Генерация — защищённый автогрессор
# -------------------------
def fully_safe_generate(
        model,
        enc_input,
        seed_dec,
        length=64,
        temperature=1.0,
        stats=None,
        sampling_mode='sample',  # 'sample' or 'argmax'
        normed=True,  # True если модель оперирует нормализованными step/dur (0..1)
        step_clip=(1e-3, 1.0),
        dur_clip=(1e-3, 4.0),
        smooth_alpha=0.2,
        verbose=False
):
    """
    Autoregressive генерация с защитой:
      - стабилизирует softmax
      - клиппит и исправляет отрицательные/аномальные step/dur
      - поддерживает sampling (temperature) и argmax режимы
    Возвращает normalized sequence shape (1, length+1, 3) — первый токен = seed_dec.
    Если normed=True -> step/dur в [0,1] (потом денормализуй с помощью stats).
    """
    enc_input = np.asarray(enc_input)
    assert enc_input.shape[0] == 1, "enc_input должен быть batch_size=1 для генерации"

    gen_list = [np.asarray(seed_dec, dtype=np.float32)]  # list of arrays (1, t, 3)
    prev_step_real = None

    # интерпретация clipping в нормализованных единицах:
    if normed and stats is not None:
        step_max_norm = 1.0
        dur_max_norm = 1.0
        # если хотим использовать реальный максимум, можно поделить: но предполагаем, что обучение было нормализовано
    else:
        # если данные уже в реальных единицах, используем переданные clip
        step_max_norm = step_clip[1]
        dur_max_norm = dur_clip[1]

    for t in range(length):
        dec_input = np.concatenate(gen_list, axis=1)  # (1, t+1, 3)
        preds = model.predict([enc_input, dec_input], verbose=0)
        if isinstance(preds, (list, tuple)) and len(preds) == 3:
            pitch_logits, step_pred, dur_pred = preds
        else:
            raise ValueError("Model.predict должен возвращать (pitch_logits, step_pred, dur_pred)")

        # берем последний timestep
        pitch_logits_last = pitch_logits[:, -1, :]  # shape (1, num_classes)
        step_last = step_pred[:, -1, 0]  # shape (1,)
        dur_last = dur_pred[:, -1, 0]  # shape (1,)

        # ---------- PITCH ----------
        probs = safe_softmax(pitch_logits_last / float(max(1e-9, temperature)))
        probs = probs.astype(np.float64)
        # защитим от tiny sums / NaN
        probs = np.nan_to_num(probs, nan=1e-8)
        probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-12)

        if sampling_mode == 'argmax':
            pitch_idx = np.argmax(probs, axis=-1)  # shape (1,)
        else:
            # sampling
            try:
                pitch_idx = [np.random.choice(probs.shape[-1], p=probs[0])]
                pitch_idx = np.array(pitch_idx, dtype=np.int32)
            except Exception:
                # fallback to argmax
                pitch_idx = np.array([int(np.argmax(probs, axis=-1)[0])], dtype=np.int32)

        # ---------- STEP / DURATION ----------
        # step_last, dur_last — предполагаем нормализованные предсказания (если normed True)
        s = float(step_last[0]) if np.ndim(step_last) else float(step_last)
        d = float(dur_last[0]) if np.ndim(dur_last) else float(dur_last)

        # Если модель предсказывает в логит/не-нормал. шкале — возможно потребуется apply activation.
        # Мы просто клиппим в разумные границы и сглаживаем.
        # Применяем clip и smoothing (в норм. шкале)
        s_fixed, d_fixed = clip_and_fix_step_dur(
            s, d,
            step_min=max(1e-6, step_clip[0]),
            step_max=step_clip[1],
            dur_min=max(1e-6, dur_clip[0]),
            dur_max=dur_clip[1],
            smooth_alpha=smooth_alpha,
            prev_step=prev_step_real
        )

        prev_step_real = s_fixed

        # формируем next token (batch 1)
        pitch_token = np.array(pitch_idx, dtype=np.float32).reshape(1, 1, 1)  # integer index as float
        step_token = np.array([[[s_fixed]]], dtype=np.float32)  # shape (1,1,1)
        dur_token = np.array([[[d_fixed]]], dtype=np.float32)

        next_token = np.concatenate([pitch_token, step_token, dur_token], axis=-1)  # (1,1,3)
        gen_list.append(next_token)

        if verbose and (t % 16 == 0):
            print(f"gen t={t}: pitch={int(pitch_idx[0])}, step={s_fixed:.4f}, dur={d_fixed:.4f}")

    generated = np.concatenate(gen_list, axis=1)  # (1, length+1, 3)
    return generated


# -------------------------
# 3) ONSET / TIMELINE helpers
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
# 4) MIDI saving (music21)
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

# -------------------------
# 5) Пример использования
# -------------------------
# Пример демонстрации всех шагов (псевдо-код, адаптируй под свои функции денормализации):
#
# enc_input = enc_inputs_ch[0:1]          # (1, L_enc, F_enc)
# seed_dec = np.zeros((1,1,3), dtype=float)
# generated_norm = fully_safe_generate(model, enc_input, seed_dec, length=64, temperature=1.0,
#                                      stats=stats, sampling_mode='sample', normed=True)
#
# # Если твоя модель выдавала нормированные step/dur в диапазоне [0,1] — денормализуй:
# # denormalize_sequence_global должен брать (1,T,3) и stats и вернуть реальные pitch/step/dur
# generated_denorm = denormalize_sequence_global(generated_norm, stats)  # ты у себя уже имеешь
#
# # Получаем numpy (T,3) без seed:
# gen_seq = generated_denorm[0, 1:, :]
#
# # enc1 и enc2 получаем из enc_inputs_ch: если у тебя enc_inputs_ch хранит реальные (ненорм) значения,
# # то их можно взять так (пример):
# # enc1_seq = enc_inputs_ch_flattened_first_voice ... (должен быть (T,3))
#
# # Выровнять timeline:
# enc1_seq, enc2_seq, gen_seq_aligned = align_timelines(enc1_seq, enc2_seq, gen_seq, method='clip')
#
# # Сохранить MIDI:
# save_multi_track_midi(enc1_seq, enc2_seq, gen_seq_aligned, fp="three_voice.mid")
