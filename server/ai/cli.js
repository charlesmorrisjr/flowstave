'use strict';
// Generate phrases from the command line (used by ai/evaluate.py).
//   node ai/cli.js --dir <folder with model.onnx> --level 5 --n 20
// Prints one JSON phrase per line.
const path = require('path');
const { Model } = require('./generate');

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
(async () => {
  const dir = path.resolve(arg('dir', path.join(__dirname, '..', 'models', 'chopin')));
  const level = Number(arg('level', 5)), n = Number(arg('n', 10));
  const m = new Model(dir);
  for (let i = 0; i < n; i++) {
    try { process.stdout.write(JSON.stringify(await m.generate(level)) + '\n'); }
    catch (e) { process.stderr.write(`level ${level}: ${e.message}\n`); }
  }
})().catch(e => { console.error(e.message); process.exit(1); });
