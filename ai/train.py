"""Train the Chopin model.

  python train.py                      # recommended "base" model on your graphics card
  python train.py --preset small       # smaller and faster
  python train.py --resume             # continue a run that was stopped (Ctrl+C is safe)

Checkpoints and logs go to out/<run>/ (best.pt = lowest validation loss).
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

from fsai import tokens
from fsai.model import Config, MusicGPT, PRESETS

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "chopin")  # set from --style in main()


class Seqs:
    """Token sequences stored back to back in a .bin file."""
    def __init__(self, split):
        self.tok = np.fromfile(os.path.join(DATA, f"{split}.bin"), dtype=np.uint16)
        self.idx = np.load(os.path.join(DATA, f"{split}_idx.npy"))

    def __len__(self):
        return len(self.idx)

    def batch(self, ids, device):
        lens = self.idx[ids, 1]
        T = int(lens.max())
        x = np.zeros((len(ids), T), dtype=np.int64)
        for r, i in enumerate(ids):
            s, n = self.idx[i]
            x[r, :n] = self.tok[s:s + n]
        x = torch.from_numpy(x)
        inp, tgt = x[:, :-1].clone(), x[:, 1:].clone()
        tgt[:, :4] = tokens.PAD  # STYLE/LEVEL/KEY/METER are given at generation time, not predicted
        return inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)


def length_batches(lens, batch, rng):
    """One epoch of batches where each batch holds sequences of similar length (little padding).
    Sequences are shuffled, sorted by length within large pools, cut into batches, and the
    batches are shuffled again, so training order stays random."""
    order = rng.permutation(len(lens))
    pool = batch * 64
    batches = []
    for i in range(0, len(order), pool):
        chunk = order[i:i + pool]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        batches.extend(chunk[j:j + batch] for j in range(0, len(chunk), batch))
    rng.shuffle(batches)
    return batches


def pick_device(want):
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():  # AMD ROCm builds of PyTorch also use the "cuda" name
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="chopin", help="which prepared data to use (data/<style>)")
    ap.add_argument("--run", default=None, help="name of this training run (default: the style)")
    ap.add_argument("--preset", default="base", choices=PRESETS)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (overrides --epochs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--no-sdpa", action="store_true", help="use plain attention (if fused attention fails on your setup)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    global DATA
    DATA = os.path.join(ROOT, "data", args.style)
    args.run = args.run or args.style
    if not os.path.exists(os.path.join(DATA, "train.bin")):
        sys.exit(f"No training data in {DATA}. Run: python prepare_data.py")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = pick_device(args.device)
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    prec = args.precision
    if prec == "auto":
        prec = "fp16" if dev.type == "cuda" else "fp32"
    print(f"Device: {name}  |  precision: {prec}")

    train, val = Seqs("train"), Seqs("val")
    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    out = os.path.join(ROOT, "out", args.run)
    os.makedirs(out, exist_ok=True)
    ckpt_last = os.path.join(out, "last.pt")

    cfg = Config(vocab_size=len(tokens.VOCAB), max_len=meta["max_len"], dropout=args.dropout, **PRESETS[args.preset])
    state = None
    if args.resume and os.path.exists(ckpt_last):
        state = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        cfg = Config(**state["config"])
        print(f"Resuming from step {state['step']}")
    model = MusicGPT(cfg, sdpa=not args.no_sdpa).to(dev)
    print(f"Model: {args.preset}, {model.n_params() / 1e6:.1f}M parameters; "
          f"{len(train):,} training sequences, {len(val):,} validation")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = args.max_steps or int(args.epochs * steps_per_epoch)
    warm = min(500, total // 10 + 1)
    lr_at = lambda s: args.lr * (s / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / max(1, total - warm)))))

    use_amp = prec in ("fp16", "bf16") and dev.type == "cuda"
    amp_dtype = torch.float16 if prec == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16" and dev.type == "cuda"))

    step, best = 0, float("inf")
    if state:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        step, best = state["step"], state["best"]
        if "scaler" in state and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg.to_dict(), "step": step, "best": best, "preset": args.preset}, path)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses, n = 0.0, 0
        sub = np.arange(min(len(val), 2048))
        sub = sub[np.argsort(val.idx[sub, 1], kind="stable")]  # similar lengths together: less padding
        for i in range(0, len(sub), args.batch):
            x, y = val.batch(sub[i:i + args.batch], dev)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            losses += float(loss) * len(x); n += len(x)
        model.train()
        return losses / max(1, n)

    def show_sample():
        model.eval()
        prompt = [tokens.BOS, tokens.TOK["STYLE_chopin"], tokens.TOK["LEVEL_free"], tokens.TOK["KEY_-3"], tokens.TOK["METER_3_4"]]
        seq = model.sample(prompt, temperature=0.9)
        ph = tokens.decode(seq)
        model.train()
        n_notes = sum(len(e["notes"]) for h in ("rh", "lh") for e in ph[h])
        ok = ph["bars"] == 4 and seq[-1] == tokens.EOS
        return f"sample: {ph['bars']} bars, {n_notes} notes, {'complete' if ok else 'incomplete'}"

    log_path = os.path.join(out, "log.csv")
    new_log = not os.path.exists(log_path) or not args.resume
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "lr", "seconds"])
    json.dump({"config": cfg.to_dict(), "preset": args.preset, "args": vars(args)}, open(os.path.join(out, "run.json"), "w"), indent=1)

    print(f"Training for {total:,} steps (~{total / steps_per_epoch:.1f} epochs). Press Ctrl+C to stop safely.\n")
    model.train()
    t0, last_t, run_loss, run_n = time.time(), time.time(), 0.0, 0
    rng = np.random.default_rng(args.seed + step)
    lens = train.idx[:, 1]
    batches, bp = length_batches(lens, args.batch, rng), 0
    try:
        while step < total:
            if bp >= len(batches):
                batches, bp = length_batches(lens, args.batch, rng), 0
            x, y = train.batch(batches[bp], dev); bp += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                print("Loss became NaN/inf. Try: python train.py --resume --precision fp32")
                break
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            step += 1
            run_loss += float(loss.detach()); run_n += 1
            if step % 50 == 0:
                rate = 50 / max(1e-6, time.time() - last_t)
                last_t = time.time()
                eta = (total - step) / max(rate, 1e-6)
                print(f"step {step:>7,}/{total:,}  loss {run_loss / run_n:.3f}  lr {lr_at(step):.2e}  "
                      f"{rate:.1f} steps/s  ETA {eta / 3600:.1f} h", flush=True)
            if step % args.eval_every == 0 or step == total:
                vl = evaluate()
                tl = run_loss / max(1, run_n); run_loss, run_n = 0.0, 0
                log.writerow([step, f"{tl:.4f}", f"{vl:.4f}", f"{lr_at(step):.2e}", int(time.time() - t0)]); logf.flush()
                mark = ""
                if vl < best:
                    best = vl; save(os.path.join(out, "best.pt")); mark = "  (best so far, saved)"
                save(ckpt_last)
                print(f"--- validation loss {vl:.3f}{mark}  |  {show_sample()}", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    save(ckpt_last)
    logf.close()
    print(f"\nSaved. Best validation loss {best:.3f}. Next: python export_onnx.py --run {args.run}")


if __name__ == "__main__":
    main()
