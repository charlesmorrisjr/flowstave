# Flowstave

A sight-reading trainer for piano. Two grand staves of music appear at a time. Notes turn green or red as you play them on a MIDI keyboard, and finished staves are replaced with new music. Ten levels lead from simple right-hand melodies to intermediate pieces, plus a Free level with no limits.

Play your keyboard into a PC and practise on an iPad or any browser on your home network. The music can come from built-in rules, a Yiruma-style rule set, or AI models trained on Chopin or on the whole Romantic piano repertoire. AI staves continue one another, and Listen plays them with expressive dynamics, rubato and pedalling.

## The two parts

```
flowstave/
├── server/   the app: run it every time you practise
└── ai/       the training kit: use it to build or improve AI music models
```

| | `server/` | `ai/` |
|---|---|---|
| What it does | Reads your MIDI keyboard, serves the app to your iPad, generates music | Downloads scores, trains models, exports them for the server |
| Language | Node.js | Python (PyTorch) |
| Needed for practising | Yes | No: only to create models |
| Setup | `npm install`, then `npm start` | Python 3.12 environment; see `ai/README.md` |
| Guide | `server/README.md` | `ai/README.md` |

**How they connect:** the training kit exports a model (`model.onnx` + `model.json`), and `python install_model.py` copies it into `server/models/<name>/`. Restart the server and the model appears in the app's Music menu. Nothing else is shared between the two parts, so the server doesn't need Python installed.

## Quick start

**Practise (server only):**
1. Install Node.js 18 or newer.
2. In `server/`, run `npm install`, then `npm start`.
3. Open the "On your iPad" address it shows, on your iPad or any browser on the same network.

The Music menu offers Built-in rules and Yiruma-style rules straight away, plus two tiny test models (Chopin and Romantic) that show how it works; real models come from the training kit.

**Train a model (once, then whenever you want to improve it):**
Follow `ai/README.md`. In short: set up Python 3.12 with AMD's PyTorch build, then in `ai/` run `get_data.py`, `prepare_data.py`, `train.py`, `export_onnx.py`, `evaluate.py` and `install_model.py`. Add `--style romantic` to the first five for the Romantic model. If you trained a model with the previous version, see "Updating from the previous version" there.

## Licences

For personal, non-commercial use. See `ai/README.md` for the training data's licences, and `server/README.md` for the piano samples (Salamander Grand Piano, CC BY 3.0) and the music font (Leland, SIL Open Font License).
