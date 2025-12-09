"""
Генерация последовательностей и конвертация в music21 объекты.
"""
from src.constants import DEVICE

import numpy as np
import torch


@torch.no_grad()
def generate_full_autoregressive(
        model, enc_input, seed_dec,
        length=64,
        pitch_temp=1.0,
        cont_temp=0.0,
        cont_scale=None,
        sampling_mode='stochastic',
        stats=None,
        device="cpu"
):
    """
    model(enc_input, dec_input) -> pitch_logits, step_pred, dur_pred
    enc_input:  (1, L_enc, F_enc)
    seed_dec:   (1, 1, 3)
    """

    model.eval()

    generated = [torch.tensor(seed_dec, dtype=torch.float32, device=device)]
    prev_step = torch.tensor(0.0, device=device)

    # cont_scale
    if cont_scale is None and stats is not None:
        cont_scale = {
            'step': float(stats.get('step_max', 1.0)),
            'dur': float(stats.get('dur_max', 1.0))
        }
    if cont_scale is None:
        cont_scale = {'step': 1.0, 'dur': 1.0}

    enc_input = torch.tensor(enc_input, dtype=torch.float32, device=device)

    for t in range(length):

        dec_input = torch.cat(generated, dim=1)  # (1, t+1, 3)

        pitch_logits, step_pred, dur_pred = model(enc_input, dec_input)

        # --- PITCH ---
        logits = pitch_logits[:, -1, :]  # (1, num_classes)

        if sampling_mode == "argmax" or pitch_temp == 0.0:
            pitch_idx = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / pitch_temp, dim=-1)
            pitch_idx = torch.multinomial(probs, num_samples=1).squeeze(1)

        pitch_next = pitch_idx.float().view(1, 1, 1)

        # --- STEP & DUR ---
        step_val = step_pred[:, -1, 0].view(1, 1, 1)
        dur_val = dur_pred[:, -1, 0].view(1, 1, 1)

        # Если NaN — fallback
        step_val = torch.nan_to_num(step_val, nan=0.25)
        dur_val = torch.nan_to_num(dur_val, nan=0.25)

        # Добавляем шум
        if sampling_mode == "stochastic" and cont_temp > 0.0:
            step_val += torch.randn_like(step_val) * (cont_temp * cont_scale['step'])
            dur_val += torch.randn_like(dur_val) * (cont_temp * cont_scale['dur'])

        step_val = torch.clamp(step_val, min=0.0)
        dur_val = torch.clamp(dur_val, min=0.0)

        step_val = step_val + prev_step
        prev_step = prev_step + step_val

        next_token = torch.cat([pitch_next, step_val, dur_val], dim=-1)
        generated.append(next_token)

    return torch.cat(generated, dim=1).cpu().numpy()


# --------------------
#  Autoregressive generation
# --------------------
@torch.no_grad()
def generate_autoregressive(model, seed, length=128, pitch_temp=1.0, cont_temp=0.0, device=DEVICE):
    """
    seed: np.array shape (1,1,3) normalized (pitch label offset, step_norm, dur_norm)
    returns numpy (1, length+1, 3) normalized (first token = seed)
    """
    model.eval()
    gen = [torch.tensor(seed, dtype=torch.float32, device=device)]

    for t in range(length):
        dec_in = torch.cat(gen, dim=1)  # (1, t+1, 3)
        pitch_logits, step_out, dur_out = model(dec_in)
        logits = pitch_logits[:, -1, :]  # (1, C)
        if pitch_temp == 0.0:
            pitch_idx = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / float(pitch_temp), dim=-1)
            probs = probs.cpu().numpy()[0]
            # sample
            pitch_idx = np.random.choice(len(probs), p=probs)
            pitch_idx = torch.tensor([pitch_idx], device=device)

        pitch_next = pitch_idx.float().view(1, 1, 1)

        step_val = step_out[:, -1, :].view(1, 1, 1)
        dur_val = dur_out[:, -1, :].view(1, 1, 1)

        # stochastic cont sampling: add gaussian noise scaled by cont_temp
        if cont_temp > 0.0:
            step_val = step_val + torch.randn_like(step_val) * cont_temp
            dur_val = dur_val + torch.randn_like(dur_val) * cont_temp

        # safety clamp
        step_val = torch.clamp(step_val, min=0.0)
        dur_val = torch.clamp(dur_val, min=0.01)

        next_token = torch.cat([pitch_next, step_val, dur_val], dim=-1)
        gen.append(next_token)

    arr = torch.cat(gen, dim=1).cpu().numpy()
    return arr  # shape (1, L+1, 3)


@torch.no_grad()
def generate_target_from_melody_encdec(model, melody_enc_norm, seed=None, length=None, pitch_temp=0.9, cont_temp=0.02,
                                       device=DEVICE):
    """
    melody_enc_norm: (1, L_enc, 3) normalized
    seed: (1,1,3) or None
    returns numpy (1, L_generated+1, 3) normalized (first token = seed if provided)
    """
    model.eval()
    enc_in = torch.tensor(melody_enc_norm, dtype=torch.float32, device=device)
    if seed is None:
        dec_tokens = [torch.zeros((1, 1, 3), dtype=torch.float32, device=device)]
    else:
        dec_tokens = [torch.tensor(seed, dtype=torch.float32, device=device)]
    target_len = length if length is not None else (melody_enc_norm.shape[1])
    for t in range(target_len):
        dec_in = torch.cat(dec_tokens, dim=1)
        pitch_logits, step_out, dur_out = model(enc_in, dec_in)
        logits = pitch_logits[:, -1, :]
        probs = torch.softmax(logits / float(pitch_temp), dim=-1)
        idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        pitch_next = idx.float().view(1, 1, 1)
        step_next = step_out[:, -1, :].view(1, 1, 1)
        dur_next = dur_out[:, -1, :].view(1, 1, 1)
        if cont_temp > 0:
            step_next = step_next + torch.randn_like(step_next) * cont_temp
            dur_next = dur_next + torch.randn_like(dur_next) * cont_temp
        step_next = torch.clamp(step_next, min=0.0)
        dur_next = torch.clamp(dur_next, min=0.01)
        next_token = torch.cat([pitch_next, step_next, dur_next], dim=-1)
        dec_tokens.append(next_token)
    return torch.cat(dec_tokens, dim=1).cpu().numpy()


@torch.no_grad()
def generate_autoregressive_decoder_only(
        model, seed, length=1,
        pitch_temp=1.0, cont_temp=0.0, device=DEVICE):
    model.eval()

    gen = torch.tensor(seed, dtype=torch.float32, device=device)  # (1, L0, 3)
    prev_step = gen[:, -1, 1].clone()  # последний step в seed (абсолютный)

    for t in range(length):
        L = gen.size(1)
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device),
            diagonal=1
        )

        pitch_logits, step_out, dur_out = model(gen, tgt_mask=causal_mask)

        # последний токен
        logits = pitch_logits[:, -1, :]

        if pitch_temp == 0.0:
            idx = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / pitch_temp, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(1)

        pitch_next = idx.float().view(1, 1, 1)

        # Δstep
        step_delta = step_out[:, -1, 0].view(1, 1, 1)
        dur_val = dur_out[:, -1, 0].view(1, 1, 1)

        step_delta = torch.clamp(step_delta, 0.0, 1.0)
        dur_val = torch.clamp(dur_val, 0.01, 1.0)

        # convert Δstep → absolute step
        step_abs = prev_step + step_delta
        prev_step = step_abs

        next_token = torch.cat([pitch_next, step_abs, dur_val], dim=-1)

        gen = torch.cat([gen, next_token], dim=1)

    return gen.cpu().numpy()
