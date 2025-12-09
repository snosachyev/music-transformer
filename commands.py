#!/usr/bin/env python3
"""
train_and_harmonize.py — module-friendly, stage-wise pipeline.

Usage:
  python train_and_harmonize.py prepare   # parse MIDI -> cache/samples.pkl
  python train_and_harmonize.py stats     # compute stats -> cache/stats.json
  python train_and_harmonize.py build     # build prepared items -> cache/prepared_items.pkl
  python train_and_harmonize.py train_melody   # train melody decoder using cached prepared items
  python train_and_harmonize.py generate  # generate example (requires trained models)

The file is defensive: if a stage fails it logs and exits without deleting caches.
"""
import pickle
import argparse

from src.constants import CACHE_DIR, ROOT
from src.stages import (
    stage_prepare, stage_stats, stage_build_items, stage_train_melody,
    generate_stage
)
from src.logger import log


# ---------------------------------------------------------
# CLI: orchestrate stages
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "stats", "build", "train_melody", "generate_melody"])
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--L_enc", type=int, default=256)
    parser.add_argument("--L_dec", type=int, default=64)
    parser.add_argument("--target", default="Piano")
    args = parser.parse_args()

    samples_pkl = CACHE_DIR / "samples.pkl"
    stats_json = CACHE_DIR / "stats.json"
    prepared_pkl = CACHE_DIR / "prepared_items.pkl"

    if args.stage == "prepare":
        stage_prepare(args.dataset, samples_pkl)
        return

    if args.stage == "stats":
        if not samples_pkl.exists():
            log.error("samples.pkl not found; run prepare first")
            return
        stage_stats(samples_pkl, stats_json)
        return

    if args.stage == "build":
        if not samples_pkl.exists() or not stats_json.exists():
            log.error("Need samples and stats; run prepare and stats first")
            return
        stage_build_items(samples_pkl, stats_json, prepared_pkl, L_enc=args.L_enc, L_dec=args.L_dec,
                          target_instrument=args.target)
        return

    # later stages require prepared items
    if not prepared_pkl.exists():
        log.error("prepared items not found; run build first")
        return

    with open(prepared_pkl, "rb") as f:
        prepared = pickle.load(f)
    items = prepared["items"]
    stats = prepared["stats"]
    pitch_offset = prepared["pitch_offset"]

    # ----------------------------
    # Train Melody (decoder-only)
    # ----------------------------
    if args.stage == "train_melody":
        stage_train_melody(items)
        return

    # ----------------------------
    # Generate (use cached models)
    # ----------------------------
    if args.stage == "generate_melody":
        # load melody model
        melody_ckpt = CACHE_DIR / "melody_best.pt"
        encdec_ckpt = CACHE_DIR / "encdec_best.pt"
        if not melody_ckpt.exists():
            log.error("Melody model not found. Run train_melody first.")
            return

        generate_stage(items, stats, melody_ckpt)
        return


if __name__ == "__main__":
    main()
