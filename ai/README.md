# Flowstave AI: Chopin and Romantic music models for sight-reading

This folder (`ai/`) trains small AI models and installs them into the server. The app can then generate staves in their style at all 10 levels. There are two collections:
- **Chopin:** Chopin's piano music only.
- **Romantic:** original piano music from the whole Romantic era (see [The Romantic model](#the-romantic-model)).

Each stave is written as the **continuation of the one before**, so a theme can be answered, repeated or developed from stave to stave instead of every stave starting from nothing.

The model only supplies the style. The level rules in `levels.json` are hard limits while it writes, so a level-3 stave can't contain level-7 rhythms or notes outside the range. They cover range, rhythms, chord size, key signature and accidentals.

There is also a **Free** level, after level 10. It's trained on Chopin's original, unsimplified phrases and writes the way Chopin wrote. It uses:
- triplets and sextuplets, and notes down to 32nds
- grace notes, and two independent voices in each hand
- compound meters such as 6/8, 12/8 and 6/4
- the full piano range, any accidentals, and Chopin's own preferred keys

It isn't part of the level-up path.

Built for **Windows 11 + AMD Radeon RX 9060 XT (16 GB)**, for personal use.

```
flowstave/
├── ai/        <- this folder: data, training, export
└── server/    <- the app server
```

## Updating from the previous version

This version changes the training data: each phrase now comes with the 4 bars before it (the context). **Models you trained before still work in the app**, but without continuity between staves. To upgrade your Chopin model you don't need to start over. Prepare the data again, then fine-tune your model on it; it learns the new part quickly and stops by itself:

```
python prepare_data.py
python train.py --data chopin --init-from chopin --run chopin-v2
python export_onnx.py --run chopin-v2 --data chopin
python evaluate.py --run chopin-v2 --data chopin
python install_model.py --run chopin-v2 --name chopin
```

`--name chopin` replaces the old model in the app; leave it out to keep both and compare them. Your original model stays in `out/chopin`.

The validation loss of the new data isn't comparable with your old numbers: the context makes the next notes easier to predict, so expect a lower figure.

For the Préludes model, prepare its data again too (`python prepare_data.py --include prelude --name chopin-preludes`) and fine-tune from the upgraded model with `--init-from chopin-v2`.

## What happens, start to finish

| Step | Command | Time (roughly) |
|---|---|---|
| Check the GPU works | `python check_gpu.py` | 1 minute |
| Download Chopin scores | `python get_data.py` | 1–3 minutes |
| Prepare training data | `python prepare_data.py` | 2–5 minutes |
| Train | `python train.py` | about 1–3 hours |
| Export for the server | `python export_onnx.py` | 1 minute |
| Check the results | `python evaluate.py` | 2–5 minutes |
| Install into the app | `python install_model.py` | seconds |

## Setup (once)

### 1. Update the AMD graphics driver

AMD's PyTorch for Windows needs **AMD Software: Adrenalin Edition 26.2.2 or newer**. Install it from amd.com, then restart.

### 2. Install Python 3.12

AMD's PyTorch packages are built for **Python 3.12 exactly**, so newer versions won't work. Download Python 3.12 from python.org. In the installer, tick **Add python.exe to PATH**.

Open **Command Prompt** in this folder (click the folder's address bar, type `cmd`, press Enter), then create a separate Python environment for the project:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
```

You'll see `(.venv)` at the start of the line. Run `.venv\Scripts\activate` again whenever you open a new Command Prompt for this project.

### 3. Install AMD's PyTorch (ROCm) and the other packages

These are AMD's commands for PyTorch on Windows with ROCm 7.2.1, from AMD's "Install PyTorch for Radeon GPUs (Windows)" page. If AMD has released a newer version, use the commands on that page instead.

```
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl

pip install -r requirements.txt
```

The second command can take several minutes.

### 4. Check the GPU

```
python check_gpu.py
```

You should see your RX 9060 XT and **Everything works**. If it suggests `--no-sdpa` or `--precision fp32`, add that option to every `train.py` command below.

## Train your Chopin model

```
python get_data.py
python prepare_data.py
python train.py
```

`get_data.py` saves the scores in `data/raw/chopin`, and `prepare_data.py` writes the training data to `data/chopin`. It prints how many phrases it found (about 5,400 from 117 pieces) and how many training sequences (about 450,000, or 65 million tokens), including about 57,000 original phrases for the Free level.

### More Nocturnes and Waltzes (optional)

Five Nocturnes and two Waltzes from the Mutopia Project are included: Nocturnes Op. 9 Nos. 1–3, Op. 27 No. 2 and Op. 72 No. 1, and Waltzes Op. 64 No. 1 and Op. 69 No. 2. They come in the `extra/mutopia` folder and `get_data.py` adds them automatically. For many more, add PDMX's public-domain MuseScore editions:

```
python get_data.py --pdmx
python prepare_data.py
```

- **Download size:** about 2.1 GB, once: a 225 MB index and a 1.9 GB score archive. The download resumes if interrupted.
- **What's kept:** only Chopin Nocturnes and Waltzes for solo piano, at most two versions of each piece (the best-rated), and only scores with no licence conflict. Easy or simplified arrangements and versions for other instruments are skipped.
- **What's deleted:** the archive afterwards. Use `--keep-pdmx` to keep it.
- **Quality:** these are user-made editions, so quality varies. Check `evaluate.py` and listen after training.

`train.py` prints progress every 50 steps:

```
step   4,050/20,736  loss 0.912  lr 5.4e-04  9.8 steps/s  ETA 0.5 h
--- validation loss 0.874  (best so far, saved)  |  sample: 4 bars, 31 notes, complete
```

- **loss** should fall quickly at first, then slowly.
- **validation loss** is measured on pieces the model never trains on. If it starts rising while the training loss keeps falling, the model is memorising rather than learning. `best.pt` always keeps the best version, so nothing is lost.
- **Stopping and resuming:** you can stop at any time with Ctrl+C, and continue later with `python train.py --resume`.

Useful options:

| Option | What it does |
|---|---|
| `--preset small` | Smaller, faster model (about 3M parameters) for a quick first try. |
| `--preset large` | Bigger model (about 25M). Only helps with more data; watch the validation loss. |
| `--epochs 3` | Shorter training (default 5). |
| `--batch 32` | Use if you run out of GPU memory (unlikely with 16 GB). |
| `--run name` | Keep several experiments side by side in `out/name/`. |

## Export, check and install

```
python export_onnx.py
python evaluate.py
python install_model.py
```

- **`export_onnx.py`** converts the model to ONNX, a format the server can run without PyTorch. It checks the converted model against PyTorch before saving.
- **`evaluate.py`** generates 30 staves per level using the same code as the server. This step needs Node.js and the `server` folder next to this one, with `npm install` already done there. It reports:
  - **rules met:** should be 100%.
  - **copied passages:** how often a generated melody repeats 10 or more notes of a real Chopin phrase, in any key. A few percent is normal, because scales and repeated notes recur everywhere. Much more means the model memorised too much, so train with fewer epochs.
  - **notes per bar, chords, and the share of staves with tuplets, two voices and grace notes**, plus keys and meters, for each level and for Free.
  - **carry:** how much of each stave's melody (as 4-note shapes) already appeared in the stave before it, as in the app. It's compared with the same figure for the real music. Near zero means the staves don't connect; much higher than the real figure means the model repeats itself.

  Staves are written in runs of 4 (`--chain`) with the app's default **Balanced** feel; try `--feel focused` or `--feel adventurous` to compare.
- **`install_model.py`** copies the model to `server/models/<run>/` (use `--name` to choose another folder name).

Restart the server. Its window should show `AI music models: chopin`. In the app, choose **Music: Chopin (AI)**, then press **Listen** to hear the two staves on screen.

## The Romantic model

A second model, trained on original solo piano works from the Romantic era. It knows more music than the Chopin model, so its harmony and phrasing are more varied, and it makes a strong starting point for fine-tuning.

```
python get_data.py --style romantic
python prepare_data.py --style romantic
python train.py --style romantic
python export_onnx.py --style romantic
python evaluate.py --style romantic
python install_model.py --run romantic
```

It appears as **"Romantic (AI)"** in the Music menu.

**What's in it** (about 280 pieces, 12,000 phrases, 1.4 million training sequences):
- **All of the Chopin sources.**
- **ASAP dataset (MusicXML):** Schumann (Kreisleriana, Arabeske, Toccata), Schubert (Impromptus, Moments musicaux, sonatas, the Wanderer Fantasy), Liszt (Transcendental Études, Ballade No. 2, the Sonata, Années de pèlerinage, a Hungarian Rhapsody, Mephisto Waltz), Brahms, Rachmaninoff's Préludes, Scriabin and Balakirev's Islamey. Transcriptions (Liszt's Paganini Études, Glinka's The Lark) are left out.
- **121 Mutopia Project pieces**, included in `extra/romantic` (credits and licences in `CREDITS.md` there): Schumann's Kinderszenen and Album for the Young, Schubert's Impromptus D. 935 and Marches militaires, Mendelssohn's Songs without Words, Grieg's Lyric Pieces, Tchaikovsky's The Seasons, Rachmaninoff's Préludes Op. 23, Scriabin's Préludes, Mussorgsky's Pictures at an Exhibition, Brahms' Waltzes, Liszt's Consolations, Burgmüller's Études Op. 100, Rimsky-Korsakov, Alkan, and Czerny's melodic studies Op. 840.
- **Optional, with `--pdmx`:** public-domain MuseScore editions of original piano works by Romantic composers from PDMX. This downloads about 2.1 GB once. Arrangements of orchestral works, operas and songs are filtered out by title, and only the best-rated edition of each piece is kept.

Training takes longer than Chopin alone: about 2–4 hours for the default 5 epochs on your graphics card. It saves `best.pt` whenever the validation loss improves, so you can stop early with Ctrl+C. The prepared data needs about 700 MB of disk space.

**Using it to improve the Chopin model.** A model that has learned the whole Romantic language first, then specialises, usually beats one trained on Chopin alone, because there is so little Chopin to learn from. After training the Romantic model:

```
python train.py --data chopin --init-from romantic --run chopin-romantic
python export_onnx.py --run chopin-romantic --data chopin --label "Chopin from Romantic (AI)"
python install_model.py --run chopin-romantic
```

You can do the same for one composer: `--include` matches folder and file names, for example `python prepare_data.py --style romantic --include schumann --name schumann`, then `python train.py --data schumann --init-from romantic`.

## Fine-tuning on one genre (for example, the Préludes)

Once your full Chopin model is trained, you can make a second, more focused model that keeps everything it learned and adds the character of one genre:

```
python prepare_data.py --include prelude --name chopin-preludes
python train.py --data chopin-preludes --init-from chopin
python export_onnx.py --run chopin-preludes --data chopin-preludes
python evaluate.py --run chopin-preludes --data chopin-preludes
python install_model.py --run chopin-preludes
```

- **`--include prelude`** keeps only files whose name contains "prelude". That's the 24 Préludes Op. 28 plus the Prélude Op. 45, giving about 870 phrases. The pattern can be any text or regular expression, for example `"waltz|valse"`.
- **`--name chopin-preludes`** saves this dataset separately in `data/chopin-preludes`, so your full Chopin data is untouched.
- **`--init-from chopin`** starts from your trained model (`out/chopin/best.pt`). Fine-tuning uses a gentler learning rate and checks progress every 200 steps. It first reports the base model's validation loss on the Préludes, then **stops by itself** after 6 checkpoints without improvement. Usually that takes minutes, not hours.
- The result is installed as **"Chopin Préludes (AI)"**, next to your full model in the Music menu, so you can compare them with Listen.

Want it more Prélude-like? Train longer with `--patience 12`. Hearing too many real Prélude passages (check the "copied" figure in `evaluate.py`)? Stop earlier, or use `--dropout 0.25`. You can rerun fine-tuning any time with different settings; it always starts again from your full model.

## How it works

1. **Data.** The training scores are:
   - Chopin's 24 Préludes and his Mazurkas, from Craig Sapp's Humdrum editions
   - 36 major works in MusicXML from the ASAP dataset: Ballades, Études, Scherzos, Sonatas, Polonaises, the Barcarolle and the Berceuse
   - five Nocturnes and two Waltzes from the Mutopia Project
   - optionally, more Nocturnes and Waltzes from PDMX

   MIDI files are snapped to a grid of 32nds and sextuplets. Each hand becomes one voice, because MIDI doesn't record voices reliably, and notes are held until the next one.
2. **Phrases.** Each piece is cut into overlapping 4-bar phrases. Time is counted in ticks, 24 per quarter note, so triplets, sextuplets and 32nds are exact. Each staff keeps up to two voices, separated the way the score separates them, and grace notes stay attached to the note they decorate. Within each voice, notes that start together become a chord, and a note is shortened when the next one in that voice starts.
3. **Easier versions.** From each phrase the pipeline also makes simpler versions: melody plus bass, simpler rhythm, beats only, and right hand only (level 1 only). Each simplified version is also stored with each hand moved by whole octaves into a comfortable register. This gives the easy levels Chopin-derived material.
4. **All keys.** Every phrase is copied into each key signature from 6 flats to 6 sharps, keeping correct spelling (F♯ versus G♭).
5. **Levels.** Each phrase is labelled with the easiest level whose rules it meets (`levels.json`). Original phrases are also labelled `free`, for the Free level.
6. **Tokens.** Each phrase becomes a sequence of tokens: `BOS STYLE_chopin LEVEL_free KEY_-2 METER_6_8 BAR POS_0 RH GR NOTE_37_0 DUR_12 NOTE_36_0 RH2 DUR_36 NOTE_30_0 LH DUR_12 NOTE_18_0 …`. Here RH2 is the right hand's lower voice, GR starts grace notes, and durations are in ticks.
7. **Model.** A small transformer (GPT-style, about 11M parameters) learns to predict the next token.
8. **Context.** Each phrase is also stored with the 4 bars that come before it in the piece: `… METER_6_8 CTX <previous 4 bars> NEXT <this phrase> EOS`. A fifth of them are also kept without it, so the model can write a first stave from nothing. In the app, each new stave is written as the continuation of the previous one.
9. **Generation.** In the server, the model writes one token at a time. Any token that would break the level's rules is removed before each choice. Easy levels, where real Chopin data is scarce, borrow the style of the nearest level with enough data, while the rules still hold to the level you're on.
10. **Best of several.** The server writes several candidate staves (2, 4 or 6, depending on the **AI feel** setting) and keeps the one the model itself rates most natural: the highest average probability of its notes. A candidate that only repeats the previous stave is penalised, or rejected if 3 or more of its 4 bars are identical.
11. **Free level.** The model is asked for `LEVEL_free` music, in a key and meter picked in proportion to how often Chopin used them. Only notation limits apply: the piano's range, rhythms the app can show, at most four notes per hand per chord, within a hand's reach (a tenth), and bars that add up.

What still can't be reproduced:
- quintuplets, septuplets and Chopin's long free-rhythm runs (such as 11 or 22 notes against a beat)
- notes faster than 32nds
- three or more independent voices in one staff (extra voices are merged into the lower voice)
- ties across bar lines (held notes end at the bar line)
- octave signs (8va)
- pieces in meters other than 2/4, 3/4, 4/4, 2/2, 3/2, 6/4, 3/8, 6/8, 9/8 and 12/8

Bars containing any of these are left out of the training phrases.

## Troubleshooting

- **`check_gpu.py` says no GPU:** make sure the Adrenalin driver is 26.2.2 or newer. Check you're inside `.venv` (you'll see `(.venv)` at the start of the line), and that step 3 finished without errors. Plain `pip install torch` installs a version that can't use AMD cards; you need AMD's packages from step 3.
- **The loss becomes NaN:** resume in full precision with `python train.py --resume --precision fp32`.
- **An error mentioning attention or `scaled_dot_product`:** add `--no-sdpa`.
- **The download fails:** run `python get_data.py` again; it skips files it already has.
- **The music sounds random:** the model needs more training. Check that the validation loss is well below 1.0, train longer, or try `--preset base` if you used `small`.
- **"This model uses the older token format 2"** when exporting: the model was trained with the previous version. Upgrade it as described in [Updating from the previous version](#updating-from-the-previous-version).
- **"The data … was prepared by an older version"** when training: run `prepare_data.py` again with the same options.
- **The music sounds too much like real Chopin passages:** check `evaluate.py`'s "copied" figure, then train with fewer epochs or more dropout, for example `--dropout 0.25`.

## Licences

Everything here is for personal, non-commercial use. The ASAP dataset is licensed CC BY-NC-SA 4.0. Craig Sapp's Chopin editions are free research editions; check their repositories for the exact terms. The Mutopia files are public domain, CC BY or CC BY-SA; see `extra/mutopia/CREDITS.md` and `extra/romantic/CREDITS.md` for each editor and licence. PDMX is CC BY 4.0, and its scores are public domain. The composers' music itself is in the public domain. The music font used by the app, Leland, is under the SIL Open Font License.
