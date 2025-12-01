"""
Генерация последовательностей и конвертация в music21 объекты.
"""

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
