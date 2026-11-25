"""Turning scores into 4-bar grand-staff phrases the Flowstave app can show.

Times are in ticks: 24 per quarter note, which represents sixteenths, thirty-seconds, and
triplets/sextuplets exactly.

A phrase is a plain dict:
    {"meter": 3, "fifths": -2, "bars": 4,
     "rh": [{"t": 0, "dur": 24, "v": 0, "notes": [[d, a], ...], "grace": [[d, a], ...]}, ...],
     "lh": [...]}
t is the start in ticks from the start of the phrase, dur one of DURS (ticks), v the voice within
the staff (0 = upper voice, stems up when there are two; 1 = lower voice), grace an optional list of
grace notes played just before the note. Each note is [d, a]: d is the diatonic step number
(C4 = 28) and a the alteration (-1 flat, 0 natural, +1 sharp), so the spelling is kept.

Within each voice the music is one rhythm at a time (notes that start together form a chord; a
note is shortened when the next one in the same voice starts), which is what the app engraves.
"""
import json
from fractions import Fraction

TPQ = 24  # ticks per quarter note
LETTERS = "CDEFGAB"
SEMI = [0, 2, 4, 5, 7, 9, 11]
SHARP_ORDER = [3, 0, 4, 1, 5, 2, 6]  # F C G D A E B (letter indexes)
FLAT_ORDER = [6, 2, 5, 1, 4, 0, 3]   # B E A D G C F
PLAIN_DURS = [96, 72, 48, 36, 24, 18, 12, 9, 6, 3]      # whole ... dotted 16th, 16th, 32nd
TRIPLET_DURS = [32, 16, 8, 4, 2]                         # triplet half, quarter, eighth, 16th, 32nd
DURS = sorted(PLAIN_DURS + TRIPLET_DURS, reverse=True)
D_MIN, D_MAX = 5, 56   # A0 .. C8
MAX_CHORD = 4
MAX_GRACE = 6
# Time signatures the app can show. A phrase's "meter" is the time signature as text, e.g. "6/8".
METERS = ["2/4", "3/4", "4/4", "2/2", "3/2", "6/4", "3/8", "6/8", "9/8", "12/8"]


def bar_ticks(meter):
    n, d = map(int, meter.split("/"))
    return n * 4 * TPQ // d


def is_triplet(d):
    return d % 3 != 0


def d_of_name(name):
    """'C4' -> 28"""
    return int(name[1:]) * 7 + LETTERS.index(name[0])


def midi_of(d, a):
    return 12 * (d // 7 + 1) + SEMI[d % 7] + a


def key_alters(fifths):
    alt = [0] * 7
    order = SHARP_ORDER if fifths > 0 else FLAT_ORDER
    for i in range(abs(fifths)):
        alt[order[i]] = 1 if fifths > 0 else -1
    return alt


def respell(d, a):
    """Bring an alteration into -1..+1 (F double-sharp -> G, etc.)."""
    while a > 1:
        m = midi_of(d, a); d += 1; a = m - midi_of(d, 0)
    while a < -1:
        m = midi_of(d, a); d -= 1; a = m - midi_of(d, 0)
    return d, a


def spell_in_key(d, a, fifths):
    """Re-spell a pitch to suit the key (MIDI files don't record F sharp versus G flat)."""
    m, ka = midi_of(d, a), key_alters(fifths)
    cands = []
    for nd in (d - 1, d, d + 1):
        na = m - midi_of(nd, 0)
        if abs(na) <= 1:
            cands.append((nd, na))
    for nd, na in cands:
        if na == ka[nd % 7]:
            return nd, na
    pref = 1 if fifths >= 0 else -1
    for nd, na in cands:
        if na == pref:
            return nd, na
    return cands[0] if cands else (d, a)


def quantize_down(ticks, allowed=DURS):
    for v in allowed:
        if ticks >= v:
            return v
    return None


# ---------------------------------------------------------------- score parsing
def _ts_meter(ts):
    m = f"{ts.numerator}/{ts.denominator}"
    return m if m in METERS else None


def score_bars(score, from_midi=False):
    """Return bars: {"start", "meter", "fifths", "valid", "raw": {"rh": [...], "lh": [...]}}.
    raw notes are (pos_ticks, voice_key, d, a, dur_ticks, is_grace, order)."""
    parts = list(score.parts)
    bars = _two_staff_bars(parts) if len(parts) == 2 else _generic_bars(score)
    if bars is None:
        return None
    if from_midi:
        if not any(b["fifths_known"] for b in bars):
            try:
                f = score.analyze("key").sharps
            except Exception:
                f = 0
            for b in bars:
                b["fifths"] = f
        # MIDI files don't record voices reliably (the importer guesses), so each hand is one voice.
        for b in bars:
            b["midi"] = True
            for hand in ("rh", "lh"):
                b["raw"][hand] = [(p, "") + spell_in_key(d, a, b["fifths"]) + (q, g, o) for p, vk, d, a, q, g, o in b["raw"][hand]]
    return bars


def _new_bar(start, meter_q, fifths, known, valid):
    return {"start": start, "meter": meter_q, "fifths": fifths, "fifths_known": known, "valid": valid,
            "raw": {"rh": [], "lh": []}}


def _generic_bars(score):
    from music21 import meter
    flat = score.flatten()
    end = float(flat.highestTime)
    if end <= 0:
        return None
    tss = [(float(t.offset), t) for t in flat.getElementsByClass("TimeSignature")] or [(0.0, meter.TimeSignature("4/4"))]
    kss = [(float(k.offset), k.sharps) for k in flat.getElementsByClass("KeySignature")]
    bars, start = [], 0.0
    while start < end - 1e-6 and len(bars) < 5000:
        cand = [t for o, t in tss if o <= start + 1e-6]
        ts = cand[-1] if cand else tss[0][1]
        ks = [k for o, k in kss if o <= start + 1e-6]
        meter_q = _ts_meter(ts)
        bars.append(_new_bar(start, meter_q, ks[-1] if ks else 0, bool(ks), meter_q is not None))
        start += float(ts.barDuration.quarterLength)
    _place(bars, score, None)
    return bars


def _two_staff_bars(parts):
    def mean_pitch(part):
        ps = [p.ps for n in part.flatten().notes for p in n.pitches]
        return sum(ps) / len(ps) if ps else 0
    if mean_pitch(parts[0]) < mean_pitch(parts[1]):
        parts = parts[::-1]
    rh_part, lh_part = parts
    end_rh, end_lh = rh_part.flatten().highestTime, lh_part.flatten().highestTime
    if min(end_rh, end_lh) <= 0 or abs(end_rh - end_lh) > 0.1 * max(end_rh, end_lh):
        return None  # parts don't line up: skip the file
    measures = list(rh_part.getElementsByClass("Measure"))
    if not measures:
        return None
    ts0 = rh_part.flatten().getElementsByClass("TimeSignature")
    ks0 = rh_part.flatten().getElementsByClass("KeySignature")
    cur_ts = ts0[0] if len(ts0) else None
    cur_fifths = ks0[0].sharps if len(ks0) else 0
    bars = []
    for m in measures:
        if m.timeSignature is not None:
            cur_ts = m.timeSignature
        if m.keySignature is not None:
            cur_fifths = m.keySignature.sharps
        meter_q = _ts_meter(cur_ts) if cur_ts is not None else None
        length = float(m.duration.quarterLength)
        valid = meter_q is not None and abs(length * TPQ - bar_ticks(meter_q)) < 1e-6 and abs(cur_fifths) <= 7
        bars.append(_new_bar(float(m.offset), meter_q, cur_fifths, len(ks0) > 0, valid))
    _place(bars, rh_part, "rh")
    _place(bars, lh_part, "lh")
    return bars


def _place(bars, container, hand):
    """Put notes into bars by absolute time, remembering which voice each came from.
    hand=None: split into hands at middle C."""
    import bisect
    from music21 import stream
    starts = [b["start"] for b in bars]
    for order, n in enumerate(container.recurse().notes):
        site = n.activeSite
        vkey = str(site.id) if isinstance(site, stream.Voice) else ""
        try:
            off = Fraction(n.getOffsetInHierarchy(container)).limit_denominator(240)
        except Exception:
            continue
        i = bisect.bisect_right(starts, float(off) + 1e-9) - 1
        if i < 0:
            continue
        bar = bars[i]
        pos = (off - Fraction(bar["start"]).limit_denominator(240)) * TPQ
        grace = bool(n.duration.isGrace)
        q = Fraction(n.duration.quarterLength).limit_denominator(240) * TPQ
        for nn in (n.notes if n.isChord else [n]):
            if nn.tie is not None and nn.tie.type in ("stop", "continue"):
                continue  # continuation of an earlier note, not a new key press
            if pos.denominator != 1:
                bar["valid"] = False  # quintuplets, septuplets and other groupings off the grid
                continue
            p = nn.pitch
            alter = int(round(p.accidental.alter)) if p.accidental is not None else 0
            d, a = respell(p.octave * 7 + LETTERS.index(p.step), alter)
            h = hand or ("rh" if p.ps >= 60 else "lh")
            bar["raw"][h].append((int(pos), vkey, d, a, float(q), grace, order))


def _voices_of(raw):
    """Map voice keys to 0 (upper) / 1 (lower) for one staff in one bar."""
    keys = {}
    for p, vk, d, a, q, g, o in raw:
        if not g:
            keys.setdefault(vk, []).append(d)
    if len(keys) <= 1:
        return {k: 0 for k in keys} or {"": 0}
    ranked = sorted(keys, key=lambda k: -sum(keys[k]) / len(keys[k]))
    return {k: (0 if i == 0 else 1) for i, k in enumerate(ranked)}


def _chordify_bar(raw, bar_ticks, legato=False):
    """Raw notes of one staff in one bar -> events, or None if not representable.
    legato: (MIDI sources) a note lasting at least half the gap to the next one is held until it."""
    vmap = _voices_of(raw)
    by_voice = {0: {}, 1: {}}
    graces = {}
    for p, vk, d, a, q, g, o in sorted(raw, key=lambda x: x[6]):
        if p < 0 or p >= bar_ticks:
            return None
        v = vmap.get(vk, 0)
        if g:
            graces.setdefault((v, p), []).append([d, a])
        else:
            by_voice[v].setdefault(p, []).append((d, a, q))
    events = []
    for v, onsets in by_voice.items():
        positions = sorted(onsets)
        for i, pos in enumerate(positions):
            nxt = positions[i + 1] if i + 1 < len(positions) else bar_ticks
            held = min(x[2] for x in onsets[pos])
            if legato and held >= (nxt - pos) / 2:
                held = nxt - pos
            dur = quantize_down(min(held, nxt - pos))
            if dur is None:
                return None  # faster than the grid allows
            seen, notes = set(), []
            for d, a, _ in sorted(onsets[pos]):
                m = midi_of(d, a)
                if m not in seen:
                    seen.add(m)
                    notes.append([d, a])
            if len(notes) > MAX_CHORD:  # keep the bass note and the top three
                notes = notes[:1] + notes[-(MAX_CHORD - 1):]
            e = {"t": pos, "dur": dur, "v": v, "notes": notes}
            gr = graces.pop((v, pos), None) or graces.pop((1 - v, pos), None)
            if gr:
                e["grace"] = gr[-MAX_GRACE:]
            events.append(e)
    if by_voice[1] and not by_voice[0]:  # a lone lower voice is just the voice
        for e in events:
            e["v"] = 0
    return sorted(events, key=lambda e: (e["t"], e["v"]))


def bar_events(bar):
    if not bar["valid"]:
        return None
    bt = bar_ticks(bar["meter"])
    rh = _chordify_bar(bar["raw"]["rh"], bt, bar.get("midi", False))
    lh = _chordify_bar(bar["raw"]["lh"], bt, bar.get("midi", False))
    if rh is None or lh is None:
        return None
    return rh, lh


def piece_phrases(bars, stride=2, length=4):
    """Slide a 4-bar window over the piece; keep windows of valid bars in one meter and key."""
    evs = [bar_events(b) for b in bars]
    out = []
    for i in range(0, len(bars) - length + 1, stride):
        win = bars[i:i + length]
        if any(e is None for e in evs[i:i + length]):
            continue
        if len({b["meter"] for b in win}) != 1 or len({b["fifths"] for b in win}) != 1:
            continue
        bt = bar_ticks(win[0]["meter"])
        ph = {"meter": win[0]["meter"], "fifths": win[0]["fifths"], "bars": length, "rh": [], "lh": []}
        empty_rh = 0
        for k, (rh, lh) in enumerate(evs[i:i + length]):
            if not rh:
                empty_rh += 1
            for hand, lst in (("rh", rh), ("lh", lh)):
                for e in lst:
                    ph[hand].append(dict(e, t=k * bt + e["t"]))
        if empty_rh > 1 or not ph["rh"]:
            continue
        out.append(ph)
    return out


# ---------------------------------------------------------------- simplified versions
def _bar_ticks(ph):
    return bar_ticks(ph["meter"])


def _refill(events, ph, allowed=DURS):
    """One voice: each note lasts until the next one (or the bar end)."""
    bt = _bar_ticks(ph)
    events = sorted(events, key=lambda e: e["t"])
    out = []
    for i, e in enumerate(events):
        bar_end = (e["t"] // bt + 1) * bt
        nxt = events[i + 1]["t"] if i + 1 < len(events) else ph["bars"] * bt
        dur = quantize_down(min(nxt, bar_end) - e["t"], allowed)
        if dur is None:
            continue
        out.append({"t": e["t"], "dur": dur, "v": 0, "notes": [list(n) for n in e["notes"]]})
    return out


def _merge(events, pick=None, grid=1, max_notes=MAX_CHORD):
    """Merge voices: all notes starting together form one chord. pick: 'top', 'bottom' or None."""
    at = {}
    for e in events:
        if e["t"] % grid == 0:
            at.setdefault(e["t"], []).extend(e["notes"])
    out = []
    for t in sorted(at):
        ns = sorted({tuple(n) for n in at[t]})
        if pick == "top":
            ns = ns[-1:]
        elif pick == "bottom":
            ns = ns[:1]
        elif len(ns) > max_notes:
            ns = ns[:1] + ns[-(max_notes - 1):]
        out.append({"t": t, "notes": [list(n) for n in ns]})
    return out


def one_voice(ph):
    """Both voices of each staff merged into chords (full texture, simpler notation; no grace notes)."""
    return dict(ph, rh=_refill(_merge(ph["rh"]), ph), lh=_refill(_merge(ph["lh"]), ph))


def melody_bass(ph):
    """Top note of the right hand; lowest left-hand note on each beat."""
    return dict(ph, rh=_refill(_merge(ph["rh"], "top"), ph), lh=_refill(_merge(ph["lh"], "bottom", grid=TPQ), ph))


def simple_rhythm(ph):
    """Only notes starting on the eighth-note grid; at most two notes per chord; plain durations."""
    return dict(ph, rh=_refill(_merge(ph["rh"], grid=12, max_notes=2), ph, PLAIN_DURS),
                lh=_refill(_merge(ph["lh"], grid=12, max_notes=2), ph, PLAIN_DURS))


def beats_only(ph):
    return dict(ph, rh=_refill(_merge(ph["rh"], "top", grid=TPQ), ph, PLAIN_DURS),
                lh=_refill(_merge(ph["lh"], "bottom", grid=TPQ), ph, PLAIN_DURS))


def right_hand_only(ph):
    return dict(ph, lh=[])


def fit_octaves(ph, rh_target=32, lh_target=21):
    """Move each hand by whole octaves so it centres near A4 (right) and C3 (left)."""
    out = dict(ph)
    for hand, target in (("rh", rh_target), ("lh", lh_target)):
        ds = [n[0] for e in ph[hand] for n in e["notes"]]
        if not ds:
            continue
        k = 7 * round((target - sum(ds) / len(ds)) / 7)
        if any(not (D_MIN <= d + k <= D_MAX) for d in ds):
            k = 0
        out[hand] = [dict(e, notes=[[d + k, a] for d, a in e["notes"]]) for e in ph[hand]]
    return out


def variants(ph):
    """The original phrase plus simplified versions (for the easier levels)."""
    mb, bo = melody_bass(ph), beats_only(ph)
    simple = [one_voice(ph), mb, simple_rhythm(ph), simple_rhythm(mb), right_hand_only(simple_rhythm(mb)), bo, right_hand_only(bo)]
    out = [ph] + simple + [fit_octaves(v) for v in simple[1:]]
    return [v for v in out if v["rh"]]


# ---------------------------------------------------------------- transposition
def transpose(ph, target_fifths):
    """Move a phrase to another key signature, keeping correct spelling. None if it won't fit."""
    k = target_fifths - ph["fifths"]
    ds, cs = (4 * k) % 7, (7 * k) % 12
    if cs > 6:
        cs -= 12
        ds -= 7

    def tp(notes):
        out = []
        for d, a in notes:
            nd = d + ds
            na = midi_of(d, a) + cs - midi_of(nd, 0)
            if abs(na) > 1:
                nd, na = respell(nd, na)
            if not (D_MIN <= nd <= D_MAX):
                return None
            out.append([nd, na])
        return out

    res = {"meter": ph["meter"], "fifths": target_fifths, "bars": ph["bars"]}
    for hand in ("rh", "lh"):
        evs = []
        for e in ph[hand]:
            ns = tp(e["notes"])
            if ns is None:
                return None
            ne = dict(e, notes=sorted(ns))
            if e.get("grace"):
                g = tp(e["grace"])
                if g is None:
                    return None
                ne["grace"] = g
            evs.append(ne)
        res[hand] = evs
    return res


# ---------------------------------------------------------------- difficulty
def load_levels(path):
    with open(path, encoding="utf-8") as f:
        return prepare_levels(json.load(f))


def prepare_levels(spec):
    for lv in spec["levels"] + ([spec["free"]] if "free" in spec else []):
        lv["rhD"] = [d_of_name(x) for x in lv["rh"]]
        lv["lhD"] = [d_of_name(x) for x in lv["lh"]] if lv["lh"] else None
        lv["durSet"] = set(lv["durs"])
        plain = [d for d in lv["durs"] if not is_triplet(d)]
        lv["grids"] = [min(plain)] + sorted({d for d in lv["durs"] if is_triplet(d)})
    return spec


def accidental_count(ph):
    ka = key_alters(ph["fifths"])
    return sum(1 for h in ("rh", "lh") for e in ph[h] for d, a in e["notes"] if a != ka[d % 7])


def fits_level(ph, lv):
    if abs(ph["fifths"]) > lv["maxFifths"] or accidental_count(ph) > lv["maxAcc"]:
        return False
    if ph["meter"] not in lv.get("meters", METERS):
        return False
    bt = _bar_ticks(ph)
    for hand, rng, mc in (("rh", lv["rhD"], lv["maxChord"][0]), ("lh", lv["lhD"], lv["maxChord"][1])):
        evs = ph[hand]
        if rng is None:
            if evs:
                return False
            continue
        for e in evs:
            if e["dur"] not in lv["durSet"] or len(e["notes"]) > mc:
                return False
            if e.get("v", 0) >= lv.get("voices", 1) or (e.get("grace") and not lv.get("grace")):
                return False
            if not any((e["t"] % bt) % g == 0 for g in lv["grids"]):
                return False
            if e["notes"][0][0] < rng[0] or e["notes"][-1][0] > rng[1]:
                return False
    if not lv["handsTogether"]:
        rt = {e["t"] for e in ph["rh"]}
        if any(e["t"] in rt for e in ph["lh"]):
            return False
    return True


def phrase_level(ph, spec):
    """The easiest level whose rules this phrase meets (11 = beyond level 10)."""
    for lv in spec["levels"]:
        if fits_level(ph, lv):
            return lv["level"]
    return len(spec["levels"]) + 1
