"""Tokenizer (format 3): phrase <-> token sequence.

    BOS STYLE_x LEVEL_n KEY_f METER_m  [CTX <the 4 bars before> NEXT]
    BAR  POS_p  RH [GR NOTE ...] DUR_x NOTE_d_a ...   RH2 ...   LH ...   LH2 ...   POS_p ...
    ... EOS

Format 3 adds the optional context: the 4 bars that came just before, so the model learns how music
continues (a theme answered, repeated or developed) instead of writing every stave from nothing.
New tokens are added at the end of the vocabulary, so format 2 models keep their token numbers.

METER_6_8 means 6/8. POS_p is the start in ticks (24 per quarter) within the bar. RH/LH mark the upper voice of each
staff, RH2/LH2 the lower voice. GR introduces grace notes played just before the note. Notes in a
chord are in ascending order. Rests are not tokens: gaps become rests when the phrase is shown.
"""
from .music import DURS, D_MIN, D_MAX, METERS, bar_ticks

FORMAT = 3
STYLES = ["chopin", "yiruma", "beethoven", "mozart", "debussy", "custom"]  # format 2 order (don't reorder)
NEW_STYLES = ["romantic"]
STYLE_CHOICES = STYLES + NEW_STYLES
N_LEVELS = 11
MAX_POS = 144  # longest bar: 12/8, 6/4, 3/2
HAND_TOKENS = [("rh", 0, "RH"), ("rh", 1, "RH2"), ("lh", 0, "LH"), ("lh", 1, "LH2")]


def build_vocab():
    v = ["PAD", "BOS", "EOS"]
    v += [f"STYLE_{s}" for s in STYLES]
    v += [f"LEVEL_{i}" for i in range(1, N_LEVELS + 1)] + ["LEVEL_free"]
    v += [f"KEY_{f}" for f in range(-7, 8)]
    v += [f"METER_{m.replace('/', '_')}" for m in METERS]
    v += ["BAR", "RH", "RH2", "LH", "LH2", "GR"]
    v += [f"POS_{p}" for p in range(MAX_POS)]
    v += [f"DUR_{d}" for d in DURS]
    v += [f"NOTE_{d}_{a}" for d in range(D_MIN, D_MAX + 1) for a in (-1, 0, 1)]
    v += [f"STYLE_{s}" for s in NEW_STYLES] + ["CTX", "NEXT"]  # format 3 additions
    return v


VOCAB = build_vocab()
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = TOK["PAD"], TOK["BOS"], TOK["EOS"]
CTX, NEXT = TOK["CTX"], TOK["NEXT"]
FORMAT2_VOCAB = len(VOCAB) - len(NEW_STYLES) - 2


def fits(ph):
    """True if every note is on the piano and can be written as a token."""
    return all(f"NOTE_{d}_{a}" in TOK for h in ("rh", "lh") for e in ph[h]
               for d, a in list(e["notes"]) + list(e.get("grace") or []))


def encode(ph, level, style="chopin", ctx=None):
    """level: 1..11 or "free". ctx: the phrase just before (same key and meter), or None."""
    seq = [BOS, TOK[f"STYLE_{style}"], TOK[f"LEVEL_{level}"], TOK[f"KEY_{ph['fifths']}"], TOK["METER_" + ph["meter"].replace("/", "_")]]
    if ctx is not None:
        seq.append(CTX)
        seq.extend(encode_bars(ctx))
        seq.append(NEXT)
    seq.extend(encode_bars(ph))
    seq.append(EOS)
    return seq


def encode_bars(ph):
    seq = []
    bt = bar_ticks(ph["meter"])
    for b in range(ph["bars"]):
        seq.append(TOK["BAR"])
        onsets = {}
        for hand in ("rh", "lh"):
            for e in ph[hand]:
                if b * bt <= e["t"] < (b + 1) * bt:
                    onsets.setdefault(e["t"] - b * bt, {})[(hand, e.get("v", 0))] = e
        for pos in sorted(onsets):
            seq.append(TOK[f"POS_{pos}"])
            for hand, v, mark in HAND_TOKENS:
                e = onsets[pos].get((hand, v))
                if not e:
                    continue
                seq.append(TOK[mark])
                if e.get("grace"):
                    seq.append(TOK["GR"])
                    seq.extend(TOK[f"NOTE_{d}_{a}"] for d, a in e["grace"])
                seq.append(TOK[f"DUR_{e['dur']}"])
                seq.extend(TOK[f"NOTE_{d}_{a}"] for d, a in e["notes"])
    return seq


def decode(seq):
    """Token ids -> phrase dict (best effort; ignores anything malformed). A context part is skipped."""
    if CTX in seq and NEXT in seq:
        a, b = seq.index(CTX), seq.index(NEXT)
        if a < b:
            seq = list(seq[:a]) + list(seq[b + 1:])
    ph = {"meter": "4/4", "fifths": 0, "bars": 0, "rh": [], "lh": []}
    bar, pos, cur, mode = -1, 0, None, None
    marks = {m: (h, v) for h, v, m in HAND_TOKENS}
    for i in seq:
        t = VOCAB[i]
        if t.startswith("KEY_"):
            ph["fifths"] = int(t[4:])
        elif t.startswith("METER_"):
            ph["meter"] = t[6:].replace("_", "/")
        elif t == "BAR":
            bar += 1; cur = None
        elif t.startswith("POS_"):
            pos = int(t[4:]); cur = None
        elif t in marks:
            h, v = marks[t]
            cur = {"t": bar * bar_ticks(ph["meter"]) + pos, "dur": 24, "v": v, "notes": []}
            ph[h].append(cur); mode = "notes"
        elif t == "GR" and cur is not None:
            cur["grace"] = []; mode = "grace"
        elif t.startswith("DUR_") and cur is not None:
            cur["dur"] = int(t[4:]); mode = "notes"
        elif t.startswith("NOTE_") and cur is not None:
            _, d, a = t.split("_")
            (cur["grace"] if mode == "grace" else cur["notes"]).append([int(d), int(a)])
        elif t == "EOS":
            break
    ph["bars"] = bar + 1
    for h in ("rh", "lh"):
        ph[h] = [e for e in ph[h] if e["notes"]]
    return ph
