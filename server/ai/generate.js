'use strict';
/*
 * Generates 4-bar grand-staff phrases with a trained model (ONNX, token format 2), while enforcing
 * the difficulty level's rules as hard limits: at every step, any token that would break a rule
 * (range, allowed rhythms and tuplets, meter, chord size, voices, grace notes, key, accidentals,
 * bar length) is removed before sampling. The model supplies the style; the rules guarantee the level.
 * Level 11 is the Free level: only notation limits.
 *
 * Times inside are in ticks (24 per quarter note). Phrases sent to the app use quarter notes.
 * Models live in models/<name>/ with model.onnx and model.json (written by export_onnx.py).
 */
const fs = require('fs');
const path = require('path');

let ort = null;
try { ort = require('onnxruntime-node'); } catch (e) { ort = null; }

const TPQ = 24;
const LETTERS = 'CDEFGAB';
const SHARP_ORDER = [3, 0, 4, 1, 5, 2, 6], FLAT_ORDER = [6, 2, 5, 1, 4, 0, 3];
const dOf = name => Number(name.slice(1)) * 7 + LETTERS.indexOf(name[0]);
function keyAlters(f) {
  const a = [0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < Math.abs(f); i++) a[f > 0 ? SHARP_ORDER[i] : FLAT_ORDER[i]] = f > 0 ? 1 : -1;
  return a;
}
const barTicks = m => { const [n, d] = m.split('/').map(Number); return n * 4 * TPQ / d; };
const isTriplet = d => d % 3 !== 0;
const REST_SIZES = [48, 24, 16, 12, 8, 6, 4, 3];
const MARKS = [['rh', 0, 'RH'], ['rh', 1, 'RH2'], ['lh', 0, 'LH'], ['lh', 1, 'LH2']];
const MAX_GRACE = 6, MAX_ONSETS_PER_BAR = 24;
const ALL_METERS = ['2/4', '3/4', '4/4', '2/2', '3/2', '6/4', '3/8', '6/8', '9/8', '12/8'];

class Model {
  constructor(dir) {
    this.dir = dir;
    this.info = JSON.parse(fs.readFileSync(path.join(dir, 'model.json'), 'utf8'));
    this.name = path.basename(dir); // the folder name is the model's id
    this.label = this.info.label || this.name;
    this.cfg = this.info.config;
    this.vocab = this.info.vocab;
    this.tok = Object.fromEntries(this.vocab.map((t, i) => [t, i]));
    if (this.tok.RH2 == null || this.tok.GR == null) throw new Error('made with an older version of the training kit (ai/); please re-export it');
    const prep = lv => {
      const plain = lv.durs.filter(d => !isTriplet(d));
      return Object.assign({}, lv, {
        rhD: lv.rh.map(dOf), lhD: lv.lh ? lv.lh.map(dOf) : null,
        grids: [Math.min(...plain)].concat(lv.durs.filter(isTriplet)),
        meters: lv.meters || ALL_METERS, voices: lv.voices || 1, grace: !!lv.grace
      });
    };
    this.levels = this.info.levels.levels.map(prep);
    this.free = prep(this.info.levels.free);
    this.pos = []; this.dur = new Map(); this.note = new Map(); this.meterTok = {};
    this.vocab.forEach((t, i) => {
      if (t.startsWith('POS_')) this.pos[Number(t.slice(4))] = i;
      else if (t.startsWith('DUR_')) this.dur.set(Number(t.slice(4)), i);
      else if (t.startsWith('NOTE_')) { const [, d, a] = t.split('_'); this.note.set(i, { d: Number(d), a: Number(a) }); }
      else if (t.startsWith('METER_')) this.meterTok[t.slice(6).replace('_', '/')] = i;
    });
    this.noteIds = [...this.note.keys()];
    this.session = null;
    this.recent = [];
  }

  async load() {
    if (!this.session) this.session = await ort.InferenceSession.create(path.join(this.dir, 'model.onnx'), { executionMode: 'sequential', intraOpNumThreads: 2 });
    return this;
  }

  publicInfo() {
    return {
      name: this.name, label: this.label,
      levels: this.levels.map(l => ({ level: l.level, title: l.title })).concat([{ level: 11, title: this.free.title }]),
      trainedSteps: this.info.trained_steps, data: this.info.data
    };
  }

  condLevel(level) {
    const counts = this.info.level_counts || {};
    for (let l = level; l <= 11; l++) if ((counts[String(l)] || 0) >= 300) return l;
    return level;
  }

  weighted(obj, allow) {
    const opts = Object.entries(obj || {}).filter(([k]) => !allow || allow(k));
    if (!opts.length) return null;
    let r = Math.random() * opts.reduce((s, x) => s + x[1], 0);
    for (const [k, v] of opts) if ((r -= v) <= 0) return k;
    return opts[0][0];
  }

  pickMeter(level, lv) {
    const mb = this.info.meters_by_level || {};
    const src = level === 11 ? mb.free : mb[String(this.condLevel(level))];
    return this.weighted(src, m => lv.meters.includes(m) && this.meterTok[m] != null) || lv.meters.find(m => this.meterTok[m] != null) || '4/4';
  }

  pickKey(lv) {
    const k = this.weighted(this.info.orig_keys, f => Math.abs(Number(f)) <= lv.maxFifths);
    return k != null ? Number(k) : Math.round(Math.random() * 2 * lv.maxFifths) - lv.maxFifths;
  }

  // level 1..10, or 11 for the Free level
  async generate(level, opts = {}) {
    await this.load();
    const free = level === 11;
    const lv = free ? this.free : this.levels[Math.max(0, Math.min(this.levels.length - 1, level - 1))];
    const condTok = free ? 'LEVEL_free' : 'LEVEL_' + this.condLevel(level);
    opts = Object.assign({}, opts);  // ignore a requested key or meter this level doesn't allow
    if (opts.fifths != null && !(Math.abs(opts.fifths) <= lv.maxFifths)) delete opts.fifths;
    if (opts.meter != null && !(lv.meters.includes(opts.meter) && this.meterTok[opts.meter] != null)) delete opts.meter;
    for (let attempt = 0; attempt < 5; attempt++) {
      const fifths = opts.fifths != null ? opts.fifths : free ? this.pickKey(lv) : Math.round(Math.random() * 2 * lv.maxFifths) - lv.maxFifths;
      const meter = opts.meter || this.pickMeter(level, lv);
      const seq = await this.sample(lv, condTok, fifths, meter, opts);
      if (!seq) continue;
      const ph = this.toPhrase(seq, lv, fifths, meter);
      if (!ph) continue;
      const sig = JSON.stringify(ph.rh.map(e => e.notes.map(n => n.d)));
      if (this.recent.includes(sig)) continue;
      this.recent.push(sig); if (this.recent.length > 40) this.recent.shift();
      ph.model = this.name; ph.level = level;
      return ph;
    }
    throw new Error('could not generate a valid phrase');
  }

  /* ---------- constrained sampling ---------- */
  async sample(lv, condTok, fifths, meter, opts) {
    const { n_layer: L, n_head: H, d_model: D, max_len: maxLen } = this.cfg;
    const Dh = D / H, V = this.vocab.length, T = this.tok;
    const temp = opts.temperature || 0.95, topP = opts.topP || 0.95;
    const ka = keyAlters(fifths);
    const bt = barTicks(meter);
    const okPos = []; for (let p = 0; p < bt; p++) okPos[p] = lv.grids.some(g => p % g === 0);
    const minLH = !lv.lhD ? 0 : lv.handsTogether ? 4 : 2;  // both hands take part from level 2 up
    const prompt = [T.BOS, T['STYLE_' + (this.info.style || 'chopin')], T[condTok], T['KEY_' + fifths], this.meterTok[meter]];
    if (prompt.some(x => x == null)) throw new Error('model vocabulary does not match');

    const s = { bars: 0, pos: -1, onsets: 0, phase: 'BAR', mark: -1, hand: null, notes: [], grace: 0, acc: 0, rhNotes: 0, lhNotes: 0, rhAtPos: false, len: prompt.length };
    const allowed = new Uint8Array(V);
    const markAllowed = mi => {
      const [hand, v] = MARKS[mi];
      if (mi <= s.mark) return false;
      if (v === 1 && lv.voices < 2) return false;
      if (hand === 'lh' && (!lv.lhD || (!lv.handsTogether && s.rhAtPos))) return false;
      return true;
    };
    const mask = () => {
      allowed.fill(0);
      const nearEnd = s.len >= maxLen - 6;
      const lhBehind = minLH > 0 && s.lhNotes < Math.ceil(minLH * s.bars / 4);
      const allowPos = () => {
        if (nearEnd || s.onsets >= MAX_ONSETS_PER_BAR) return;
        for (let p = s.pos + 1; p < bt; p++) if (okPos[p]) allowed[this.pos[p]] = 1;
      };
      const allowEnd = () => { if (s.bars < 4) allowed[T.BAR] = 1; else if (s.rhNotes >= 3 && s.lhNotes >= minLH) allowed[T.EOS] = 1; };
      const allowMarks = () => {
        for (let mi = 0; mi < 4; mi++) {
          if (!markAllowed(mi)) continue;
          if (lhBehind && MARKS[mi][0] === 'rh' && !lv.handsTogether) continue;
          allowed[T[MARKS[mi][2]]] = 1;
        }
      };
      const allowNotes = (ascending) => {
        const hi = s.hand === 'rh' ? 0 : 1, rng = s.hand === 'rh' ? lv.rhD : lv.lhD;
        const last = s.notes[s.notes.length - 1], first = s.notes[0];
        for (const id of this.noteIds) {
          const n = this.note.get(id);
          if (n.d < rng[0] || n.d > rng[1]) continue;
          if (ascending && last && (n.d <= last.d || n.d - first.d > (lv.maxSpan || 7))) continue;
          if (ascending && n.a !== ka[n.d % 7] && s.acc >= lv.maxAcc) continue;
          allowed[id] = 1;
        }
        return hi;
      };
      switch (s.phase) {
        case 'BAR': allowed[T.BAR] = 1; return;
        case 'POS': allowPos(); return;
        case 'HAND': allowMarks(); return;
        case 'AFTER_MARK':
          if (lv.grace) allowed[T.GR] = 1;
          for (const d of lv.durs) if (s.pos + d <= bt) allowed[this.dur.get(d)] = 1;
          return;
        case 'GRACE': allowNotes(false); if (s.grace >= 1) for (const d of lv.durs) if (s.pos + d <= bt) allowed[this.dur.get(d)] = 1;
          if (s.grace >= MAX_GRACE) for (const id of this.noteIds) allowed[id] = 0;
          return;
        case 'NOTE': allowNotes(true); return;
        case 'MORE': {
          const hi = s.hand === 'rh' ? 0 : 1;
          if (s.notes.length < lv.maxChord[hi]) allowNotes(true);
          const mustLH = lhBehind && s.hand === 'rh' && lv.handsTogether && lv.lhD;
          for (let mi = 0; mi < 4; mi++) if (markAllowed(mi) && (!mustLH || MARKS[mi][0] === 'lh')) allowed[T[MARKS[mi][2]]] = 1;
          if (!mustLH) { allowPos(); allowEnd(); }
          return;
        }
      }
    };
    const advance = id => {
      const t = this.vocab[id];
      s.len++;
      if (t === 'BAR') { s.bars++; s.pos = -1; s.onsets = 0; s.phase = 'POS'; }
      else if (t.startsWith('POS_')) { s.pos = Number(t.slice(4)); s.onsets++; s.phase = 'HAND'; s.mark = -1; s.rhAtPos = false; }
      else if (t === 'GR') { s.phase = 'GRACE'; s.grace = 0; }
      else if (t.startsWith('DUR_')) s.phase = 'NOTE';
      else if (t.startsWith('NOTE_')) {
        if (s.phase === 'GRACE') { s.grace++; return; }
        const n = this.note.get(id); s.notes.push(n);
        if (n.a !== ka[n.d % 7]) s.acc++;
        if (s.hand === 'rh') s.rhNotes++; else s.lhNotes++;
        s.phase = 'MORE';
      } else {
        const mi = MARKS.findIndex(m => m[2] === t);
        if (mi >= 0) { s.mark = mi; s.hand = MARKS[mi][0]; s.notes = []; s.phase = 'AFTER_MARK'; if (s.hand === 'rh') s.rhAtPos = true; }
      }
    };

    let pk = new ort.Tensor('float32', new Float32Array(0), [L, 1, H, 0, Dh]);
    let pv = new ort.Tensor('float32', new Float32Array(0), [L, 1, H, 0, Dh]);
    let input = prompt.slice();
    const seq = prompt.slice();
    const probs = new Float64Array(V);
    for (let step = 0; step < maxLen - prompt.length; step++) {
      const idx = new ort.Tensor('int64', BigInt64Array.from(input.map(BigInt)), [1, input.length]);
      const out = await this.session.run({ idx, past_k: pk, past_v: pv });
      pk = out.new_k; pv = out.new_v;
      const logits = out.logits.data;
      mask();
      let mx = -Infinity;
      for (let i = 0; i < V; i++) if (allowed[i] && logits[i] > mx) mx = logits[i];
      if (mx === -Infinity) return null;
      let sum = 0;
      for (let i = 0; i < V; i++) { probs[i] = allowed[i] ? Math.exp((logits[i] - mx) / temp) : 0; sum += probs[i]; }
      const order = []; for (let i = 0; i < V; i++) if (probs[i] > 0) order.push(i);
      order.sort((a, b) => probs[b] - probs[a]);
      let acc = 0, cut = order.length;
      for (let k = 0; k < order.length; k++) { acc += probs[order[k]] / sum; if (acc >= topP) { cut = k + 1; break; } }
      let r = Math.random() * order.slice(0, cut).reduce((t, i) => t + probs[i], 0), pick = order[0];
      for (let k = 0; k < cut; k++) { r -= probs[order[k]]; if (r <= 0) { pick = order[k]; break; } }
      seq.push(pick);
      if (pick === T.EOS) return seq;
      advance(pick);
      input = [pick];
    }
    return null;
  }

  /* ---------- tokens -> staves the app can draw (with rests, per voice) ---------- */
  toPhrase(seq, lv, fifths, meter) {
    const bt = barTicks(meter);
    const raw = [];
    let bar = -1, pos = 0, cur = null, mode = null;
    for (const id of seq) {
      const t = this.vocab[id];
      if (t === 'BAR') { bar++; cur = null; }
      else if (t.startsWith('POS_')) { pos = Number(t.slice(4)); cur = null; }
      else if (t === 'GR' && cur) { cur.grace = []; mode = 'g'; }
      else if (t.startsWith('DUR_') && cur) { cur.dur = Number(t.slice(4)); mode = 'n'; }
      else if (t.startsWith('NOTE_') && cur) { const n = this.note.get(id); (mode === 'g' ? cur.grace : cur.notes).push({ d: n.d, a: n.a }); }
      else {
        const m = MARKS.find(x => x[2] === t);
        if (m) { cur = { hand: m[0], v: m[1], t: bar * bt + pos, dur: TPQ, notes: [] }; raw.push(cur); mode = 'n'; }
      }
    }
    if (bar !== 3) return null;
    const durs = lv.durs.slice().sort((a, b) => b - a);
    const qdown = x => { for (const d of durs) if (x >= d) return d; return null; };
    const out = { ts: meter, meter: bt / TPQ, fifths, bars: 4, rh: [], lh: [] };
    const toQ = e => Object.assign({}, e, { t: e.t / TPQ, dur: e.dur / TPQ });
    for (const hand of ['rh', 'lh']) {
      const all = [];
      for (const v of [0, 1]) {
        const evs = raw.filter(e => e.hand === hand && e.v === v && e.notes.length).sort((a, b) => a.t - b.t);
        for (let i = 0; i < evs.length; i++) {        // one rhythm per voice: cut overlaps
          const e = evs[i], barEnd = (Math.floor(e.t / bt) + 1) * bt;
          const lim = Math.min(barEnd, i + 1 < evs.length ? evs[i + 1].t : Infinity) - e.t;
          if (e.dur > lim) e.dur = qdown(lim);
        }
        const notes = evs.filter(e => e.dur);
        for (let b = 0; b < 4; b++) {
          const inBar = notes.filter(e => e.t >= b * bt && e.t < (b + 1) * bt);
          if (!inBar.length) { if (v === 0) all.push({ t: b * bt, dur: bt, v, rest: true, full: true, notes: [] }); continue; }
          let t = b * bt;
          const fill = upto => {
            while (upto - t > 0) {
              const rel = t - b * bt;
              const r = REST_SIZES.find(x => x <= upto - t && rel % x === 0) || REST_SIZES.find(x => x <= upto - t);
              if (!r) break;
              all.push({ t, dur: r, v, rest: true, notes: [] });
              t += r;
            }
          };
          for (const e of inBar) {
            fill(e.t);
            const ev = { t: e.t, dur: e.dur, v, rest: false, notes: e.notes };
            if (e.grace && e.grace.length) ev.grace = e.grace;
            all.push(ev); t = e.t + e.dur;
          }
          fill((b + 1) * bt);
        }
      }
      // A bar where only the lower voice has notes: show it as the single voice.
      out[hand] = all.sort((a, b) => a.t - b.t || a.v - b.v).map(toQ);
    }
    if (out.rh.filter(e => !e.rest).length < 3) return null;
    return out;
  }
}

/* ---------- model registry ---------- */
const models = new Map();
let queue = Promise.resolve();

function loadModels(dir) {
  models.clear();
  if (!ort) return { error: 'onnxruntime-node is not installed (run npm install)' };
  if (!fs.existsSync(dir)) return { names: [] };
  const skipped = [];
  for (const name of fs.readdirSync(dir)) {
    const d = path.join(dir, name);
    if (fs.existsSync(path.join(d, 'model.onnx')) && fs.existsSync(path.join(d, 'model.json'))) {
      try { const m = new Model(d); models.set(m.name, m); } catch (e) { skipped.push(`${name} (${e.message})`); }
    }
  }
  return { names: [...models.keys()], skipped };
}

function generate(name, level, count = 1, opts = {}) {
  const m = models.get(name);
  if (!m) return Promise.reject(new Error(`no model named "${name}"`));
  const job = queue.then(async () => {
    const out = [];
    let o = Object.assign({}, opts);
    for (let i = 0; i < count; i++) {
      const ph = await m.generate(level, o);
      out.push(ph);
      o = Object.assign({}, o, { fifths: ph.fifths, meter: ph.ts });  // one request = one key and meter
    }
    return out;
  });
  queue = job.catch(() => {});
  return job;
}

module.exports = { loadModels, generate, models, Model };
