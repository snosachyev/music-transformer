import numpy as np


# -----------------------------
# Tokenizer (как у тебя)
# -----------------------------
class Music21Tokenizer:
    def __init__(self, pitch_range=(21, 108), max_duration=16):
        self.min_pitch, self.max_pitch = pitch_range
        self.max_duration = max_duration

    def note_to_token(self, note):
        pitch = note['pitch'] - self.min_pitch
        pitch = np.clip(pitch, 0, self.max_pitch - self.min_pitch)
        dur = int(np.clip(note['dur'], 1, self.max_duration))
        start = int(round(note['start'] * 4))  # quantize to 16th notes
        return [pitch, start, dur]

    def tokenize_instrument(self, notes):
        tokens = [self.note_to_token(n) for n in notes]
        if tokens:
            return np.array(tokens, dtype=np.int32)
        return np.zeros((0, 3), dtype=np.int32)
