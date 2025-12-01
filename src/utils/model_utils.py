"""
Создание архитектуры Transformer (энкодер-декодер).
"""

import torch
import torch.nn as nn
import math


class TripleLoss(nn.Module):
    def __init__(self, pitch_vocab):
        super().__init__()
        self.pitch_loss = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pitch_logits, step_out, dur_out, pitch_targets, step_targets, dur_targets):
        B, L, C = pitch_logits.shape
        pitch_logits_flat = pitch_logits.reshape(B * L, C)
        pitch_targets_flat = pitch_targets.reshape(B * L)

        step_out_flat = step_out.reshape(B * L, -1)
        step_targets_flat = step_targets.reshape(B * L, -1)

        dur_out_flat = dur_out.reshape(B * L, -1)
        dur_targets_flat = dur_targets.reshape(B * L, -1)

        loss_pitch = self.pitch_loss(pitch_logits_flat, pitch_targets_flat)
        loss_step = self.mse(step_out_flat, step_targets_flat)
        loss_dur = self.mse(dur_out_flat, dur_targets_flat)

        return loss_pitch + loss_step + loss_dur, (loss_pitch, loss_step, loss_dur)


# -----------------------------------------------------
# Linear embedding: (pitch, step, dur) → d_model
# -----------------------------------------------------
class EventEmbedding(nn.Module):
    def __init__(self, d_model, pitch_vocab):
        super().__init__()
        self.pitch_emb = nn.Embedding(pitch_vocab, d_model)
        self.step_linear = nn.Linear(1, d_model)
        self.dur_linear = nn.Linear(1, d_model)
        self.proj = nn.Linear(d_model * 3, d_model)

    def forward(self, x):
        # x: (B, L, 3)
        pitch = x[..., 0].long()
        step = x[..., 1].float()
        dur = x[..., 2].float()

        pitch_emb = self.pitch_emb(pitch)
        step_emb = self.step_linear(step.unsqueeze(-1))
        dur_emb = self.dur_linear(dur.unsqueeze(-1))

        out = torch.cat([pitch_emb, step_emb, dur_emb], dim=-1)
        return self.proj(out)


# -----------------------------------------------------
# Model B — Encoder–Decoder Transformer
# -----------------------------------------------------
class MusicTransformerED(nn.Module):
    def __init__(self, pitch_vocab=128, d_model=256, n_heads=8, n_layers=6):
        super().__init__()
        self.pitch_emb = nn.Embedding(pitch_vocab, d_model)
        self.step_emb = nn.Linear(1, d_model)
        self.dur_emb = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.pitch_out = nn.Linear(d_model, pitch_vocab)
        self.step_out = nn.Linear(d_model, 1)
        self.dur_out = nn.Linear(d_model, 1)

    def embed(self, x):
        pitch = x[..., 0].long()
        step = x[..., 1:2].float()
        dur = x[..., 2:3].float()
        return self.pitch_emb(pitch) + self.step_emb(step) + self.dur_emb(dur)

    def forward(self, enc_in, dec_in, enc_mask=None, dec_mask=None):
        enc = self.embed(enc_in)
        dec = self.embed(dec_in)
        mem = self.encoder(enc, src_key_padding_mask=enc_mask)
        out = self.decoder(dec, mem, tgt_key_padding_mask=dec_mask, memory_key_padding_mask=enc_mask)
        pitch = self.pitch_out(out)
        step = self.step_out(out)
        dur = self.dur_out(out)
        return pitch, step, dur


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, :x.size(1), :]
