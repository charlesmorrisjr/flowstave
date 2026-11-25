"""Check a trained model before installing it.

  python evaluate.py --run chopin

Generates staves at every level with the same code the server uses (needs Node.js and the
server folder), then reports:
  - rule check: every stave must meet its level's rules (should be 100%)
  - copying: how often a generated melody repeats a long passage from Chopin's real pieces
    (transposition doesn't hide it; the comparison uses intervals and rhythm)
  - texture: notes per bar, chords, rests, keys and meters
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

from fsai import music

ROOT = os.path.dirname(os.path.abspath(__file__))


def default_server():
    """The server folder: ../server in the combined project (or ../flowstave-server in older layouts)."""
    parent = os.path.dirname(ROOT)
    for name in ("server", "flowstave-server"):
        if os.path.exists(os.path.join(parent, name, "server.js")):
            return os.path.join(parent, name)
    return os.path.join(parent, "server")


def melody_sig(ph):
    """Top notes of the right hand (upper voice) as (interval, time step) pairs: the same tune in any
    key matches. Expects pipeline phrases (times in ticks, notes as [d, a])."""
    evs = sorted((e for e in ph["rh"] if e.get("notes") and e.get("v", 0) == 0), key=lambda e: e["t"])
    tops = [(e["t"], music.midi_of(*e["notes"][-1])) for e in evs]
    return [(tops[i][1] - tops[i - 1][1], tops[i][0] - tops[i - 1][0]) for i in range(1, len(tops))]


def ngrams(sig, n):
    return {tuple(sig[i:i + n]) for i in range(len(sig) - n + 1)}


def as_lists(ph):
    """Server phrases (quarter notes, {d,a} objects) -> pipeline phrases (ticks, [d,a] lists)."""
    out = {"meter": ph["ts"], "fifths": ph["fifths"], "bars": 4}
    for h in ("rh", "lh"):
        out[h] = []
        for e in ph[h]:
            if e.get("rest"):
                continue
            ne = {"t": round(e["t"] * music.TPQ), "dur": round(e["dur"] * music.TPQ), "v": e.get("v", 0),
                  "notes": [[n["d"], n["a"]] for n in e["notes"]]}
            if e.get("grace"):
                ne["grace"] = [[n["d"], n["a"]] for n in e["grace"]]
            out[h].append(ne)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="chopin")
    ap.add_argument("--run", default=None, help="training run to check (default: the style)")
    ap.add_argument("--n", type=int, default=30, help="staves per level")
    ap.add_argument("--ngram", type=int, default=10, help="melody steps that count as a copied passage")
    ap.add_argument("--server", default=default_server())
    args = ap.parse_args()
    args.run = args.run or args.style

    exp = os.path.join(ROOT, "out", args.run, "export")
    cli = os.path.join(args.server, "ai", "cli.js")
    if not os.path.exists(os.path.join(exp, "model.onnx")):
        sys.exit(f"No export in {exp}. Run: python export_onnx.py --run {args.run}")
    if not os.path.exists(cli):
        sys.exit(f"Flowstave server not found at {args.server} (needed to generate). Use --server.")
    # Check against the rules the model was exported with (the ones the server enforces).
    with open(os.path.join(exp, "model.json"), encoding="utf-8") as f:
        spec = music.prepare_levels(json.load(f)["levels"])

    print("Indexing Chopin's melodies...")
    index = {}
    with open(os.path.join(ROOT, "data", args.style, "reference.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for g in ngrams(melody_sig(r["phrase"]), args.ngram):
                index.setdefault(g, r["piece"])

    total, broken, copied = 0, 0, 0
    examples = []
    print(f"\n{'level':>5} {'staves':>6} {'rules ok':>9} {'copied':>7} {'notes/bar':>9} {'chords':>7} {'tuplets':>7} {'2 voices':>8} {'grace':>5}  keys / meters")
    for lvl in list(range(1, 11)) + ["free"]:
        p = subprocess.run(["node", cli, "--dir", exp, "--level", "11" if lvl == "free" else str(lvl), "--n", str(args.n)],
                           capture_output=True, text=True)
        phrases = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
        if not phrases:
            print(f"{lvl!s:>5}  generation failed: {p.stderr.strip()[:200]}")
            continue
        ok = cp = 0
        notes = chords = tup = two = gr = 0
        keys, meters = Counter(), Counter()
        for ph in phrases:
            lp = as_lists(ph)
            rules = spec["free"] if lvl == "free" else spec["levels"][lvl - 1]
            if music.fits_level(lp, rules):
                ok += 1
            hit = [index[g] for g in ngrams(melody_sig(lp), args.ngram) if g in index]
            if hit:
                cp += 1
                examples.append((lvl, Counter(hit).most_common(1)[0][0]))
            for h in ("rh", "lh"):
                for e in lp[h]:
                    notes += len(e["notes"])
                    chords += len(e["notes"]) > 1
            tup += any(music.is_triplet(e["dur"]) for h in ("rh", "lh") for e in lp[h])
            two += any(e["v"] == 1 for h in ("rh", "lh") for e in lp[h])
            gr += any(e.get("grace") for h in ("rh", "lh") for e in lp[h])
            keys[ph["fifths"]] += 1
            meters[ph["ts"]] += 1
        n = len(phrases)
        total += n; broken += n - ok; copied += cp
        print(f"{lvl!s:>5} {n:>6} {ok / n:>8.0%} {cp / n:>7.0%} {notes / n / 4:>9.1f} {chords / n:>7.1f} {tup / n:>7.0%} {two / n:>8.0%} {gr / n:>5.0%}  "
              f"{dict(sorted(keys.items()))} / {dict(meters)}")

    print(f"\nRules met: {(total - broken) / max(1, total):.1%} of {total} staves.")
    print(f"Copied passages ({args.ngram}+ melody steps matching a real Chopin phrase): {copied / max(1, total):.1%}.")
    if examples:
        print("Most similar sources:", ", ".join(f"{p} (level {l})" for l, p in examples[:8]))
    print("\nA few percent of matches is normal (scales and repeated notes recur everywhere).\n"
          "If it is much higher, the model has memorised too much: train for fewer epochs or with more dropout.")
    print("Then listen: install with  python install_model.py --run " + args.run + "  and use Listen in the app.")


if __name__ == "__main__":
    main()
