import torch

from torch.utils.data import DataLoader
from src.constants import DEVICE, CACHE_DIR

from src.utils.dataset_utils import SimpleDecDataset
from src.utils.train_utils import EarlyStopping, train_epoch_decoder_only

from src.logger import log
from src.models.decoder_only import DecoderOnlyMusicTransformer, MelodyLoss


def stage_train_melody(items):
    # dataset: only decoder side
    dec_items = [{"dec_in": it["dec_in"], "pitch_targets": it["pitch_targets"], "step_targets": it["step_targets"],
                  "dur_targets": it["dur_targets"]} for it in items]
    ds = SimpleDecDataset(dec_items)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = DecoderOnlyMusicTransformer().to(DEVICE)
    crit = MelodyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    early = EarlyStopping(patience=4, min_delta=1e-4)
    best = CACHE_DIR / "melody_best.pt"
    for ep in range(50):
        loss = train_epoch_decoder_only(model, loader, opt, crit, DEVICE)
        log.info("[Melody] epoch %d loss %.4f", ep + 1, loss)
        early.step(loss)
        if loss <= early.best_loss + 1e-12:
            torch.save(model.state_dict(), best)
            log.info("Saved melody best -> %s", best)
        if early.should_stop:
            log.info("Melody early stop")
            break
