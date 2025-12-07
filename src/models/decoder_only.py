from src.constants import PITCH_VOCAB
import torch
import torch.nn as nn


# --------------------
#  Model: decoder-only (как обсуждали)
# --------------------
class EventEmbedding(nn.Module):
    def __init__(self, d_model, pitch_vocab):
        super().__init__()
        self.pitch_emb = nn.Embedding(pitch_vocab, d_model)
        self.step_linear = nn.Linear(1, d_model)
        self.dur_linear = nn.Linear(1, d_model)
        self.proj = nn.Linear(d_model * 3, d_model)

    def forward(self, x):
        pitch = x[..., 0].long()
        step = x[..., 1].unsqueeze(-1).float()
        dur = x[..., 2].unsqueeze(-1).float()
        out = torch.cat([self.pitch_emb(pitch), self.step_linear(step), self.dur_linear(dur)], dim=-1)
        return self.proj(out)


class DecoderOnlyMusicTransformer(nn.Module):
    def __init__(self, pitch_vocab=PITCH_VOCAB, d_model=256, n_heads=8, n_layers=6):
        super().__init__()
        self.embed = EventEmbedding(d_model, pitch_vocab)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.pitch_out = nn.Linear(d_model, pitch_vocab)
        self.step_out = nn.Linear(d_model, 1)
        self.dur_out = nn.Linear(d_model, 1)

    def forward(self, dec_in):
        """
        dec_in: (B, L, 3)
        """
        x = self.embed(dec_in)  # (B, L, d_model)
        L = dec_in.size(1)
        # causal mask (upper triangular True -> masked)
        # TransformerDecoder expects tgt_mask with shape (L, L) where True=allow? torch wants float mask with -inf? We'll use bool mask that PyTorch supports.
        causal_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=dec_in.device), diagonal=1)
        # For memory, pass x as both memory and target (self-attention via decoder with memory=x)
        out = self.decoder(x, x, tgt_mask=causal_mask)
        pitch = self.pitch_out(out)
        step = self.step_out(out)
        dur = self.dur_out(out)
        return pitch, step, dur


# --------------------
#  Loss for training (pitch cross-entropy + MSE for step/dur)
# --------------------
class MelodyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pitch_logits, step_out, dur_out, target):
        # pitch_logits: (B, L, C)
        # step_out: (B, L, 1), dur_out: (B, L, 1)
        # target: (B, L, 3) normalized (pitch, step, dur)
        B, L, C = pitch_logits.shape
        pitch_t = target[..., 0].long().reshape(B * L)
        pitch_logits_flat = pitch_logits.reshape(B * L, C)
        loss_p = self.ce(pitch_logits_flat, pitch_t)

        step_t = target[..., 1].reshape(B * L, 1)
        dur_t = target[..., 2].reshape(B * L, 1)
        step_pred = step_out.reshape(B * L, 1)
        dur_pred = dur_out.reshape(B * L, 1)

        loss_s = self.mse(step_pred, step_t)
        loss_d = self.mse(dur_pred, dur_t)

        print("pitch_t.min", pitch_t.min(), "pitch_t.max", pitch_t.max())

        print("pitch_logits.shape", pitch_logits.shape)

        return loss_p + loss_s + loss_d, (loss_p.item(), loss_s.item(), loss_d.item())


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in loader:
            dec_in = batch['dec_in'].to(device)
            dec_tgt = batch['dec_tgt'].to(device)
            pitch_logits, step_out, dur_out = model(dec_in)
            loss, parts = criterion(pitch_logits, step_out, dur_out, dec_tgt)
            total_loss += loss.item()
            total_steps += 1
    return total_loss / max(1, total_steps)
