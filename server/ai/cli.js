'use strict';
// Generate phrases from the command line (used by ai/evaluate.py).
//   node ai/cli.js --dir <folder with model.onnx> --level 5 --n 20 [--feel balanced] [--chain 4]
// Prints one JSON phrase per line. With --chain k, phrases come in runs of k staves, each continuing
// the one before (same key and meter, with context), as in the app.
const path = require('path');
const { Model } = require('./generate');

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
(async () => {
  const dir = path.resolve(arg('dir', path.join(__dirname, '..', 'models', 'chopin')));
  const level = Number(arg('level', 5)), n = Number(arg('n', 10));
  const feel = arg('feel', 'balanced'), chain = Math.max(1, Number(arg('chain', 1)));
  const m = new Model(dir);
  let o = { feel };
  for (let i = 0; i < n; i++) {
    if (i % chain === 0) o = { feel };
    try {
      const ph = await m.generate(level, o);
      ph.run = Math.floor(i / chain);
      process.stdout.write(JSON.stringify(ph) + '\n');
      o = { feel, fifths: ph.fifths, meter: ph.ts, ctx: m.parseCtx(ph.ctx, ph.fifths, ph.ts) };
    } catch (e) { process.stderr.write(`level ${level}: ${e.message}\n`); }
  }
})().catch(e => { console.error(e.message); process.exit(1); });
