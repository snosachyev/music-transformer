import torch

from src.constants import DEVICE, OUTPUT_DIR

from src.logger import log
from src.utils.generate_utils import generate_autoregressive_decoder_only
from src.models import DecoderOnlyMusicTransformer, MusicTransformerED
from src.utils.data_utils import denormalize_sequence_global
from src.utils.generate_utils import generate_target_from_melody_encdec
from src.utils.music21_utils import save_multi_track_midi


def generate_stage(items, stats, melody_ckpt, encdec_ckpt=None, length=32):
    melody_model = DecoderOnlyMusicTransformer().to(DEVICE)
    melody_model.load_state_dict(torch.load(melody_ckpt, map_location=DEVICE))
    melody_model.eval()
    # choose seed from prepared items
    seed_item = items[0]
    seed = seed_item["dec_in"][:4]  # first 4 tokens as seed
    seed = seed[None, ...]  # (1, L_seed, 3)
    log.info("Seed shape: %s", seed.shape)
    # gen melody
    gen_norm = generate_autoregressive_decoder_only(melody_model, seed, length=length, pitch_temp=0.9, cont_temp=0.02,
                                                    device=DEVICE)
    gen_den = denormalize_sequence_global(gen_norm, stats)
    if gen_den.ndim == 3 and gen_den.shape[0] == 1:
        melody_seq = gen_den[0, 1:, :]
    else:
        melody_seq = gen_den[1:, :]
    # optionally generate encdec accompaniment if model present
    if encdec_ckpt is not None and encdec_ckpt.exists():
        encdec = MusicTransformerED().to(DEVICE)
        encdec.load_state_dict(torch.load(encdec_ckpt, map_location=DEVICE))
        encdec.eval()
        # use normalized gen_norm as encoder input (1, L, 3)
        bass_norm = generate_target_from_melody_encdec(encdec, gen_norm, seed=None, length=gen_norm.shape[1] - 1,
                                                       device=DEVICE)
        bass_den = denormalize_sequence_global(bass_norm, stats)
        bass_seq = bass_den[0, 1:, :] if bass_den.ndim == 3 else bass_den[1:, :]
        # save 2-track midi (melody + bass)
        outp = OUTPUT_DIR / "generated_melody_with_bass.mid"
        save_multi_track_midi(melody_seq, bass_seq, [], fp=str(outp), instr1="Piano", instr2="AcousticBass",
                              instr_gen="Piano")
        log.info("Saved generated MIDI with bass: %s", outp)
    else:
        # save only melody
        outp = OUTPUT_DIR / "generated_melody.mid"
        save_multi_track_midi(melody_seq, [], [], fp=str(outp), instr1="Piano", instr2="", instr_gen="Piano")
        log.info("Saved melody only: %s", outp)
