import pickle

from pathlib import Path

from src.logger import log

from src.utils.music21_utils import get_all_midis, extract_sample, _safe_note_fields


# ---------------------------------------------------------
# Stage 1: prepare: parse MIDI -> list of sample dicts -> cache
# ---------------------------------------------------------
def stage_prepare(dataset_dir: str, out_pkl: Path):
    log.info("Stage PREPARE: scanning %s", dataset_dir)
    midis = get_all_midis(dataset_dir)
    log.info("Found %d MIDI files", len(midis))

    samples = []
    failed = 0

    # Собираем все инструменты
    instruments = []
    for score in midis:
        instruments.extend(detect_instruments_in_midi(score))
    global_instruments = set(instruments)

    for i, m in enumerate(midis):
        try:
            # use user's extract_sample if available
            s = extract_sample(m, global_instruments)
            # sanitize every note dict: ensure pitch/start/dur
            for inst, notes in list(s.items()):
                good = []
                for raw in notes:
                    try:
                        p, st, du = _safe_note_fields(raw)
                        if p is None:
                            continue
                        good.append({"pitch": int(p), "start": float(st), "dur": float(du)})
                    except Exception:
                        continue
                s[inst] = good
            samples.append(s)
        except Exception as e:
            log.warning("Failed to parse MIDI idx %d: %s", i, e)
            failed += 1
            continue
    log.info("Prepared %d samples, failed %d", len(samples), failed)
    with open(out_pkl, "wb") as f:
        pickle.dump(samples, f)
    log.info("Saved samples to %s", out_pkl)
    return samples
