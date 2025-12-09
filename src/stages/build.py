import json
import pickle

from pathlib import Path

from src.logger import log


# ---------------------------------------------------------
# Stage 3: build prepared items (normalize, dec_in shift) -> cache
# ---------------------------------------------------------
def stage_build_items(samples_pkl: Path, stats_json: Path, out_pkl: Path, L_enc=256, L_dec=64,
                      target_instrument="Piano"):
    log.info("Stage BUILD: building prepared items")
    with open(samples_pkl, "rb") as f:
        samples = pickle.load(f)
    with open(stats_json, "r") as f:
        stats = json.load(f)
    pitch_offset = int(stats.get("pitch_min", 0))

    prepared = []
    skipped = 0
    for s in samples:
        # require target_instrument present to build mel/dataset
        if target_instrument not in s or len(s[target_instrument]) == 0:
            skipped += 1
            continue
            # build enc_in as concatenation of all non-target instruments
            enc_notes = []
        for k, ns in s.items():
            if k == target_instrument: continue
            enc_notes.extend(ns)
        enc_notes.sort(key=lambda x: x["start"])
        enc_arr = []
        for n in enc_notes[:L_enc]:
            enc_arr.append([n["pitch"] - pitch_offset, n["start"] / stats["step_max"], n["dur"] / stats["dur_max"]])
        while len(enc_arr) < L_enc:
            enc_arr.append([0, 0, 0])
        enc_arr = np.array(enc_arr, dtype=np.float32)
        # decoder target from target_instrument
        dec_notes = s[target_instrument]
        dec_notes.sort(key=lambda x: x["start"])
        pitches, steps, durs = [], [], []
        prev_start = dec_notes[0]["start"] if len(dec_notes) > 0 else 0.0
        for n in dec_notes[:L_dec]:
            pitches.append(int(n["pitch"] - pitch_offset))
            steps.append([float(n["start"] - prev_start) / stats["step_max"]])
            durs.append([float(n["dur"]) / stats["dur_max"]])
            prev_start = n["start"]
        while len(pitches) < L_dec:
            pitches.append(0)
            steps.append([0.0])
            durs.append([0.0])
        pitch_targets = np.array(pitches, dtype=np.int64)
        step_targets = np.array(steps, dtype=np.float32)
        dur_targets = np.array(durs, dtype=np.float32)
        # dec_in = shifted targets (teacher forcing)
        dec_in = np.zeros((L_dec, 3), dtype=np.float32)
        if L_dec > 1:
            dec_in[1:, 0] = pitch_targets[:-1]
            dec_in[1:, 1] = step_targets[:-1, 0]
            dec_in[1:, 2] = dur_targets[:-1, 0]
        prepared.append({
            "enc_in": enc_arr,
            "dec_in": dec_in,
            "pitch_targets": pitch_targets,
            "step_targets": step_targets,
            "dur_targets": dur_targets
        })
    log.info("Built %d prepared items, skipped %d", len(prepared), skipped)
    with open(out_pkl, "wb") as f:
        pickle.dump({"items": prepared, "stats": stats, "pitch_offset": pitch_offset}, f)
    log.info("Saved prepared items to %s", out_pkl)
    return prepared, stats, pitch_offset
