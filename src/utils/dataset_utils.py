import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader


def collate_chunks(batch):
    """
    batch: список словарей, каждый словарь = chunk из MIDIDataset
    Возвращает словарь с батчами:
      enc_in: (batch_size, chunk_size, L_enc, 3)
      dec_in: (batch_size, chunk_size, L_dec, 3)
      pitch_targets: (batch_size, chunk_size, 1)
      step_targets: (batch_size, chunk_size, 1)
      dur_targets: (batch_size, chunk_size, 1)
    """
    enc_in = torch.stack([item['enc_in'] for item in batch], dim=0)
    dec_in = torch.stack([item['dec_in'] for item in batch], dim=0)
    pitch_targets = torch.stack([item['pitch_targets'] for item in batch], dim=0)
    step_targets = torch.stack([item['step_targets'] for item in batch], dim=0)
    dur_targets = torch.stack([item['dur_targets'] for item in batch], dim=0)

    return {
        'enc_in': enc_in,
        'dec_in': dec_in,
        'pitch_targets': pitch_targets,
        'step_targets': step_targets,
        'dur_targets': dur_targets
    }


def prepare_sample(sample,
                   input_instruments,
                   target_instrument,
                   L_enc,
                   L_dec,
                   pitch_offset,
                   step_max,
                   dur_max):
    """
    Превращает сырой sample (из extract_sample)
    в dict с numpy массивами:
      enc_in:        (L_enc, 3)
      dec_in:        (L_dec, 3)
      pitch_targets: (L_dec,)
      step_targets:  (L_dec, 1)
      dur_targets:   (L_dec, 1)
    """

    # -----------------------------
    # Encoder input: concat всех input instruments
    # -----------------------------
    enc_notes = []
    for inst in input_instruments or []:
        if inst in sample:
            enc_notes.extend(sample[inst])
    enc_notes.sort(key=lambda x: x['start'])

    # нормализация
    enc_arr = []
    for n in enc_notes[:L_enc]:
        enc_arr.append([
            n['pitch'] - pitch_offset,
            n['start'] / step_max,
            n['dur'] / dur_max
        ])

    # паддинг
    while len(enc_arr) < L_enc:
        enc_arr.append([0, 0, 0])
    enc_arr = np.array(enc_arr, dtype=np.float32)

    # -----------------------------
    # Decoder input + Targets (target instrument ONLY)
    # -----------------------------
    dec_notes = sample.get(target_instrument, [])
    dec_notes.sort(key=lambda x: x['start'])

    pitches = []
    steps = []
    durs = []

    prev_start = 0.0
    for n in dec_notes[:L_dec]:
        pitches.append(n['pitch'] - pitch_offset)
        steps.append([n['start'] - prev_start])
        durs.append([n['dur']])
        prev_start = n['start']

    # Паддинг targets
    while len(pitches) < L_dec:
        pitches.append(0)
        steps.append([0])
        durs.append([0])

    pitch_targets = np.array(pitches, dtype=np.int64)
    step_targets = np.array(steps, dtype=np.float32)
    dur_targets = np.array(durs, dtype=np.float32)

    # -----------------------------
    # Decoder input = targets but shifted (teacher forcing)
    # -----------------------------
    dec_in = []
    dec_prev_start = 0.0
    for i in range(min(L_dec, len(dec_notes))):
        n = dec_notes[i]
        dec_in.append([
            n['pitch'] - pitch_offset,
            (n['start'] - dec_prev_start) / step_max,
            n['dur'] / dur_max
        ])
        dec_prev_start = n['start']

    while len(dec_in) < L_dec:
        dec_in.append([0, 0, 0])
    dec_in = np.array(dec_in, dtype=np.float32)

    return {
        "enc_in": enc_arr,  # (L_enc, 3)
        "dec_in": dec_in,  # (L_dec, 3)
        "pitch_targets": pitch_targets,  # (L_dec,)
        "step_targets": step_targets,  # (L_dec, 1)
        "dur_targets": dur_targets  # (L_dec, 1)
    }


# -----------------------------
# Dataset
# -----------------------------
class MIDIDataset(Dataset):
    def __init__(self, samples, tokenizer, stats, pitch_offset,
                 L_enc=256, L_dec=128,
                 input_instruments=None,
                 target_instrument='Piano'):
        self.samples = samples
        self.tokenizer = tokenizer
        self.stats = stats
        self.pitch_offset = pitch_offset
        self.L_enc = L_enc
        self.L_dec = L_dec
        self.input_instruments = input_instruments
        self.target_instrument = target_instrument

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        out = prepare_sample(
            sample,
            self.input_instruments,
            self.target_instrument,
            # убираем seq_len, используем L_dec для таргетов
            L_enc=self.L_enc,
            L_dec=self.L_dec,
            pitch_offset=self.pitch_offset,
            step_max=self.stats["step_max"],
            dur_max=self.stats["dur_max"]
        )

        # Приводим к torch
        return {
            'enc_in': torch.from_numpy(out['enc_in']).float(),  # [L_enc, F]
            'dec_in': torch.from_numpy(out['dec_in']).float(),  # [L_dec, F]
            'pitch_targets': torch.from_numpy(out['pitch_targets']).long(),  # [L_dec]
            'step_targets': torch.from_numpy(out['step_targets']).float(),  # [L_dec, 1]
            'dur_targets': torch.from_numpy(out['dur_targets']).float(),  # [L_dec, 1]
        }
