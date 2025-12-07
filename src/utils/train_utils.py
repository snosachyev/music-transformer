from src.constants import DEVICE

import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        """
        patience — сколько эпох ждать улучшений
        min_delta — минимальное улучшение (чтобы считалось улучшением)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True


# --------------------
#  Training / helpers
# --------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_steps = 0
    for batch in loader:
        dec_in = batch['dec_in'].to(DEVICE)  # (B, L, 3)
        # Собираем таргет обратно в (B, L, 3) без numpy
        pitch_t = batch['pitch_targets'].unsqueeze(-1)  # (B, L, 1)
        step_t = batch['step_targets']  # (B, L, 1)
        dur_t = batch['dur_targets']  # (B, L, 1)

        dec_tgt = torch.cat([pitch_t, step_t, dur_t], dim=-1).to(DEVICE)  # (B, L, 3)

        optimizer.zero_grad()
        pitch_logits, step_out, dur_out = model(dec_in)
        loss, parts = criterion(pitch_logits, step_out, dur_out, dec_tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_steps += 1
    return total_loss / max(1, total_steps)
