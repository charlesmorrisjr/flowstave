"""Turn the downloaded scores into training data.

  1. Parse every score in data/raw (music21).
  2. Cut each piece into overlapping 4-bar phrases the app can show (2/4, 3/4 or 4/4; no tuplets
     or notes faster than sixteenths).
  3. Add simplified versions (melody + bass, simpler rhythm, right hand only) so the easy levels
     have Chopin-derived examples.
  4. Copy each phrase into all key signatures from 6 flats to 6 sharps.
  5. Label each phrase with its difficulty level (levels.json) and turn it into tokens.

Original (unsimplified) phrases are also stored with the token LEVEL_free, for the app's Free level.

Output (data/<style>/): train.bin, train_idx.npy, val.bin, val_idx.npy, meta.json, reference.jsonl
Run:  python prepare_data.py                       (Chopin, from data/raw/chopin)
      python prepare_data.py --style yiruma        (your own files in data/raw/yiruma)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fsai import music, tokens

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_LEN = 1024
EXTS = (".krn", ".musicxml", ".xml", ".mxl", ".mid", ".midi")


def parse_file(job):
    """Worker: one score -> list of phrases (or an error message)."""
    path, stride = job
    try:
        import music21
        is_midi = path.lower().endswith((".mid", ".midi"))
        if is_midi:  # snap timing to a grid of 32nds and sextuplets (keeps triplets)
            score = music21.converter.parse(path, quantizePost=True, quarterLengthDivisors=(8, 6))
        else:
            score = music21.converter.parse(path)
        bars = music.score_bars(score, from_midi=is_midi)
        if bars is None:
            return path, [], "skipped (couldn't find piano staves, or parts don't line up)"
        return path, music.piece_phrases(bars, stride=stride), None
    except Exception as e:  # a broken file shouldn't stop the whole run
        return path, [], f"error: {e}"


def is_val(path, frac):
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
    return (h % 1000) / 1000 < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="chopin", choices=tokens.STYLES)
    ap.add_argument("--raw", default=None, help="folder of scores (default data/raw/<style>)")
    ap.add_argument("--stride", type=int, default=0, help="bars between phrase windows (default 2; 1 for small collections)")
    ap.add_argument("--val", type=float, default=0.07, help="share of pieces held out for validation")
    ap.add_argument("--keys", type=int, default=6, help="transpose to key signatures -K..K")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="only use the first N files (quick tests)")
    args = ap.parse_args()

    RAW = args.raw or os.path.join(ROOT, "data", "raw", args.style)
    OUT = os.path.join(ROOT, "data", args.style)
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(RAW) for f in fs if f.lower().endswith(EXTS))
    if args.limit:
        files = files[:args.limit]
    if not files:
        hint = "Run: python get_data.py" if args.style == "chopin" else f"Put your MusicXML or MIDI files in {RAW}"
        sys.exit(f"No scores found in {RAW}. {hint}")
    stride = args.stride or (1 if len(files) < 60 else 2)
    spec = music.load_levels(os.path.join(ROOT, "levels.json"))

    print(f"Parsing {len(files)} scores with {args.workers} worker(s)...")
    t0 = time.time()
    pieces = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (path, phrases, err) in enumerate(ex.map(parse_file, [(f, stride) for f in files]), 1):
            name = os.path.relpath(path, RAW)
            if err:
                print(f"  [{i}/{len(files)}] {name}: {err}")
            else:
                print(f"  [{i}/{len(files)}] {name}: {len(phrases)} phrases")
                pieces.append((path, phrases))
    print(f"Parsed in {time.time() - t0:.0f}s")

    splits = {"train": [], "val": []}
    seen = set()
    level_counts = {"train": Counter(), "val": Counter()}
    meter_by_level = {}
    key_counts = Counter()
    orig_keys = Counter()
    ref = open(os.path.join(OUT, "reference.jsonl"), "w", encoding="utf-8")
    n_phr = 0
    for path, phrases in pieces:
        split = "val" if is_val(path, args.val) else "train"
        for ph in phrases:
            n_phr += 1
            ref.write(json.dumps({"piece": os.path.basename(path), "phrase": ph}) + "\n")
            if split == "train":
                orig_keys[ph["fifths"]] += 1
                meter_by_level.setdefault("free", Counter())[ph["meter"]] += 1
            for vi, var in enumerate(music.variants(ph)):
                for f in range(-args.keys, args.keys + 1):
                    tp = var if f == var["fifths"] else music.transpose(var, f)
                    if tp is None:
                        continue
                    lvl = music.phrase_level(tp, spec)
                    if not tp["lh"] and ph["lh"] and lvl != 1:
                        continue  # right-hand-only versions are only for level 1
                    labels = [lvl] + (["free"] if vi == 0 else [])  # originals also teach the Free level
                    for lab in labels:
                        seq = tokens.encode(tp, lab, args.style)
                        if len(seq) > MAX_LEN:
                            continue
                        key = hashlib.md5(np.asarray(seq, dtype=np.uint16).tobytes()).digest()
                        if key in seen:
                            continue
                        seen.add(key)
                        splits[split].append(seq)
                        level_counts[split][lab] += 1
                        if split == "train" and lab != "free":
                            meter_by_level.setdefault(lab, Counter())[tp["meter"]] += 1
                            key_counts[tp["fifths"]] += 1
    ref.close()

    for split, seqs in splits.items():
        if not seqs:
            continue
        idx = np.zeros((len(seqs), 2), dtype=np.int64)
        flat = np.concatenate([np.asarray(s, dtype=np.uint16) for s in seqs])
        pos = 0
        for i, s in enumerate(seqs):
            idx[i] = (pos, len(s))
            pos += len(s)
        flat.tofile(os.path.join(OUT, f"{split}.bin"))
        np.save(os.path.join(OUT, f"{split}_idx.npy"), idx)

    lens = [len(s) for s in splits["train"]]
    meta = {
        "vocab": tokens.VOCAB,
        "format": tokens.FORMAT,
        "style": args.style,
        "orig_keys": {str(k): v for k, v in sorted(orig_keys.items())},
        "pieces": len(pieces),
        "phrases": n_phr,
        "sequences": {k: len(v) for k, v in splits.items()},
        "tokens": {k: int(sum(len(s) for s in v)) for k, v in splits.items()},
        "max_len": MAX_LEN,
        "mean_len": float(np.mean(lens)) if lens else 0,
        "level_counts": {str(k): v for k, v in level_counts["train"].items()},
        "val_level_counts": {str(k): v for k, v in level_counts["val"].items()},
        "meters_by_level": {str(k): {str(m): c for m, c in v.items()} for k, v in meter_by_level.items()},
        "keys": {str(k): v for k, v in sorted(key_counts.items())},
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"\n{len(pieces)} pieces -> {n_phr} phrases -> {meta['sequences']['train']} training and "
          f"{meta['sequences'].get('val', 0)} validation sequences ({meta['tokens']['train']:,} training tokens, "
          f"average {meta['mean_len']:.0f} tokens).")
    print("Training sequences per level:", ", ".join(f"{k}: {v}" for k, v in meta["level_counts"].items()))
    nxt = "python train.py" if args.style == "chopin" else f"python train.py --style {args.style} --init-from chopin"
    print(f"\nDone. Next: {nxt}")


if __name__ == "__main__":
    main()
