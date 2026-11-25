# Flowstave: sight-reading trainer with MIDI over your home network

Play your MIDI keyboard into your PC and practise on your iPad. The PC reads the keyboard and serves the app. The iPad shows the music in Safari and grades every note.

```
MIDI keyboard ──USB──> PC (Flowstave server) ──Wi-Fi──> iPad (Safari)
```

The PC records the exact time of every note as it arrives, and the iPad keeps its clock in sync with the PC's. So Wi-Fi delay only affects how soon a note turns green or red. It doesn't change whether the note counts as on time.

## What you need

- A PC or Mac with **Node.js 18 or newer** (download the LTS version from nodejs.org).
- A MIDI keyboard or digital piano connected to that computer by USB.
- An iPad (or any tablet, phone or laptop) on the **same Wi-Fi network**.

## Setup (once)

1. Unzip this folder somewhere convenient, such as Documents.
2. Open a terminal in the folder:
   - Windows: open the folder, click the address bar, type `cmd` and press Enter.
   - Mac: right-click the folder in Finder and choose **New Terminal at Folder**.
3. Install the two dependencies:

   ```
   npm install
   ```

## Every time you practise

1. Plug in the keyboard. (It can also be plugged in later; the server checks every couple of seconds.)
2. Start the server: run `npm start` in the folder.
3. The window shows addresses like this:

   ```
   On this computer:  http://localhost:3000
   On your iPad:      http://192.168.1.20:3000
   ```

4. On the iPad, open Safari and go to the **On your iPad** address.
5. The status line under the music should show your keyboard's name and the network delay, for example `MIDI: Digital Piano · network 4 ms`.

Optional: in Safari, tap Share > **Add to Home Screen** to open Flowstave full screen like an app.

Keep the server window open while you practise. Close it, or press Ctrl+C, to stop.

## Tips for the lowest delay

- Connect the PC to your router with an Ethernet cable if you can.
- Use your router's 5 GHz Wi-Fi for the iPad, and turn off Low Power Mode.
- Your keyboard makes its own sound, so leave **Sound for MIDI keyboard** off.
- If you use the metronome, play it through the iPad's speaker or wired headphones. Bluetooth headphones lag by 150–250 ms.
- Give the PC a fixed IP address in your router settings, so the iPad bookmark never changes.

## Keeping the iPad screen on

Safari only lets a page keep the screen awake over HTTPS. Over plain `http://` the app shows a tip instead. You have two options:

- **Simple:** on the iPad, go to Settings > Display & Brightness > Auto-Lock and choose **Never** while you practise.
- **HTTPS (one-time setup):**
  1. Install [mkcert](https://github.com/FiloSottile/mkcert) on the PC and run `mkcert -install`.
  2. In this folder, run (with your PC's own address):

     ```
     mkdir certs
     mkcert -key-file certs/key.pem -cert-file certs/cert.pem 192.168.1.20 localhost
     ```

  3. Copy `rootCA.pem` (run `mkcert -CAROOT` to find its folder) to the iPad, for example by AirDrop. Install it in Settings > General > VPN & Device Management. Then turn it on in Settings > General > About > Certificate Trust Settings.
  4. Restart the server. It switches to `https://` automatically. Open the new `https://` address on the iPad and **Keep screen awake** will appear.

## AI music (Chopin)

The **Music** menu under the sheet music chooses where new staves come from:

- **Built-in rules:** the original generator.
- **Yiruma-style (rules):** modern lyrical piano in the style of Yiruma, made entirely by rules. It needs no model or training, and works at every level and in Free. It uses:
  - pop chord progressions (I–V–vi–IV and others), with staves paired as question (ending on V) and answer (ending on I)
  - left hand that grows with the level: rests, taking turns, held bass notes, broken fifths, eighth-note arpeggios up to the tenth, add-9 colours, then flowing sixteenths
  - a singable melody built from a motif and its repeat, with leaning notes (appoggiaturas) and syncopation from level 6
  - thirds and sixths under the melody at higher levels; at level 10 and in Free, octave climaxes, an inner voice and grace notes
  - warm keys such as A, D, E, A♭ and E♭ major, limited by what each level allows
- **Chopin (AI)**, or any other installed model: staves written by a model trained on Chopin. The current level's rules still apply exactly.

Models live in `models/<name>/` (`model.onnx` + `model.json`). To create and install one, follow `ai/README.md` in the project, which ends with `python install_model.py`. Restart the server after installing a model; its window lists the models it found.

**Key and time signature** stay the same for every stave until you press **New music**. Changing level, including a level-up, or changing the Music menu also starts new music in a new key, because each level allows different keys. Switching between Wait for me and Tempo, or pressing Stop, keeps them.

**Free** comes after level 10 on the level arrows. With an AI model, it writes in the model's natural style, with no level limits: full range, any key and accidentals, and chords of up to four notes. It isn't part of the level-up path: your accuracy still shows, but you stay on Free. With built-in rules, Free plays level-10 music.

The app shows everything the Chopin model writes:
- triplets and sextuplets, with their numbers, and notes down to 32nds
- grace notes, drawn small before the note
- two voices in one staff, with stems up and down
- compound meters such as 6/8 and 12/8, with beats grouped in threes
- clef changes when a staff's music sits high or low
- very dense staves are drawn a little smaller to fit

When grading, grace notes are optional: playing them is never counted as a mistake. A key struck by both voices at once is graded once. In tempo mode with 6/8, 9/8 or 12/8, the metronome clicks on each dotted-quarter beat.

The server writes a few staves ahead, so you never wait. If a stave isn't ready in time, that one stave comes from the built-in rules.

Press **Listen** to hear the two staves on screen, starting with the one the cursor is on. Playback uses the BPM shown next to the button.

## Piano sound

Listen, and the sound for on-screen and computer keys, use a sampled concert grand: the **Salamander Grand Piano** (a Yamaha C5 recorded by Alexander Holm, licensed CC BY 3.0). `npm install` downloads two velocity layers, about 11 MB, and the server serves them, so no internet is needed while you practise.

- **Melody and accompaniment:** the melody plays from the louder layer and the accompaniment from the softer one, so the right hand sings over the left.
- **Loading:** the samples load the first time you tap the page, which takes a second or two on home Wi-Fi.
- **Pedal when listening:** each note rings until the end of its bar, as a pianist would pedal. Untick it for a dry, exact sound.
- **Piano: Simple synth** switches back to the old synthesized sound.

## Settings

Set these before `npm start`. On Mac/Linux use `PORT=3001 npm start`; on Windows use `set PORT=3001` then `npm start`.

| Variable | Default | What it does |
|---|---|---|
| `PORT` | `3000` | Port the server listens on. |
| `HOST` | `0.0.0.0` | Network address to listen on (all by default). |
| `MIDI_FILTER` | (none) | Only use MIDI inputs whose name contains this text, e.g. `MIDI_FILTER=yamaha`. |

## Troubleshooting

- **The iPad can't open the page.** Check that both devices are on the same Wi-Fi. Windows may ask whether to allow Node.js on networks the first time; choose **Private networks**. If you missed that, allow Node.js in Windows Defender Firewall. On a Mac, allow incoming connections for `node` if asked.
- **"No MIDI keyboard on the PC yet."** Check the USB cable and that the keyboard is on. On Windows, only one program can use a MIDI keyboard at a time, so close your DAW or any other music app.
- **Several MIDI devices are connected and you only want one.** Use `MIDI_FILTER`.
- **"Port 3000 is already in use."** Another program is using that port. Start with a different `PORT`.
- **Notes register but sound late on the iPad.** That's the iPad's audio output, not the network. Turn off the iPad sound for MIDI and use the keyboard's own sound.
- **Grading feels a bit strict or loose.** Change **Timing** (Relaxed / Normal / Strict) under the music.

## What's in the folder

- `server.js`: reads MIDI, records the time of each note, sends notes to connected browsers, answers clock-sync messages, and serves the app.
- `public/index.html`: the app (music generation, notation, grading, levels).
- `ai/generate.js`: runs AI models and enforces the level rules while they write. `ai/cli.js` generates from the command line.
- `models/`: installed AI models.
- Piano samples come from the npm packages `@audio-samples/piano-mp3-velocity8` and `-velocity13` (Salamander Grand Piano V3, CC BY 3.0, Alexander Holm).

Several browsers can connect at once; they all receive the same notes.
