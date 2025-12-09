import torch.nn as nn


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
