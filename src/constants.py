import random
import numpy as np
import torch


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
# Use utils denorm if available
# denormalize_sequence_global = (denorm_from_utils if has_utils else denormalize_sequence_global_local)

input_instruments = ['Piaro right', 'Piano left2']
target_instrument = 'Piano'

pitch_offset = 21
rare_note = None
SEQ_LEN = 32
ENC_LEN = 256
DEC_LEN = 128
