# ---------------------------------------------------------
# Stage 2: stats: compute global stats -> cache
# ---------------------------------------------------------
import json
import pickle

from pathlib import Path

from src.logger import log
from src.utils.music21_utils import compute_global_stats, get_rare_note


def stage_stats(samples_pkl: Path, out_json: Path):
    log.info("Stage STATS: computing global stats from %s", samples_pkl)
    with open(samples_pkl, "rb") as f:
        samples = pickle.load(f)
    notes = []
    for s in samples:
        for inst, ns in s.items():
            for n in ns:
                p = n.get("pitch", None)
                st = n.get("start", 0.0)
                du = n.get("dur", 0.5)
                if p is None:
                    continue
                notes.append({"pitch": int(p), "step": float(st), "dur": float(du)})
    if len(notes) == 0:
        raise RuntimeError("No notes to compute stats.")

    rare_notes = get_rare_note(notes)
    # **Удаление редких нот**

    for element in notes:
        if element['pitch'] in rare_notes:
            notes.remove(element)

    stats = compute_global_stats(notes)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Saved stats to %s: %s", out_json, stats)
    return stats
