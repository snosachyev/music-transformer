import random
import sys
import numpy as np
import torch

from pathlib import Path


# --------------------
#  Параметры
# --------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 50
BATCH_SIZE = 32
LR_PRETRAIN = 3e-4
LR_FINETUNE = 2e-5

SEQ_LEN = 64  # длина входной последовательности (включая последующий таргет shift)
GEN_LENGTH = 128  # сколько токенов генерировать
PITCH_VOCAB = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------------------------------------------------------
# Basic config
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

input_instruments = ['Piaro right', 'Piano left2']
target_instrument = 'Piano'

pitch_offset = 21
rare_note = None
SEQ_LEN = 32
ENC_LEN = 256
DEC_LEN = 128
