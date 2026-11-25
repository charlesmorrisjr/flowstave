"""Export a trained model for the Flowstave server.

  python export_onnx.py --run chopin

Writes out/<run>/export/: model.onnx, model.json (settings, vocabulary, levels, data statistics).
The export is checked against PyTorch before it is saved.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from fsai import tokens
from fsai.model import Config, MusicGPT

ROOT = os.path.dirname(os.path.abspath(__file__))


class Step(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, idx, past_k, past_v):
        return self.m.forward_cached(idx, past_k, past_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="chopin")
    ap.add_argument("--run", default=None, help="training run to export (default: the style)")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--label", default="Chopin (AI)")
    args = ap.parse_args()
    args.run = args.run or args.style

    run = os.path.join(ROOT, "out", args.run)
    path = os.path.join(run, args.checkpoint)
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Train first: python train.py --run {args.run}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**state["config"])
    model = MusicGPT(cfg, sdpa=False)
    model.load_state_dict(state["model"])
    model.eval()
    step = Step(model).eval()

    L, H, Dh = cfg.n_layer, cfg.n_head, cfg.d_model // cfg.n_head
    idx = torch.tensor([[tokens.BOS, tokens.TOK["STYLE_chopin"], tokens.TOK["LEVEL_6"], tokens.TOK["KEY_0"], tokens.TOK["METER_4_4"]]])
    pk = torch.zeros(L, 1, H, 0, Dh)
    pv = torch.zeros(L, 1, H, 0, Dh)

    exp = os.path.join(run, "export")
    os.makedirs(exp, exist_ok=True)
    onnx_path = os.path.join(exp, "model.onnx")
    dyn = {"idx": {1: "T"}, "past_k": {3: "P"}, "past_v": {3: "P"}, "new_k": {3: "PT"}, "new_v": {3: "PT"}}
    kw = dict(input_names=["idx", "past_k", "past_v"], output_names=["logits", "new_k", "new_v"], opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(step, (idx, pk, pv), onnx_path, dynamic_axes=dyn, dynamo=False, **kw)
        except TypeError:  # older PyTorch without the dynamo argument
            torch.onnx.export(step, (idx, pk, pv), onnx_path, dynamic_axes=dyn, **kw)

    # Check: run the ONNX model step by step and compare with PyTorch on the whole sequence.
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    seq = idx[0].tolist()
    k, v = pk.numpy(), pv.numpy()
    x = idx.numpy()
    worst = 0.0
    rng = np.random.default_rng(0)
    for i in range(40):
        logits, k, v = sess.run(None, {"idx": x.astype(np.int64), "past_k": k, "past_v": v})
        with torch.no_grad():
            ref = model(torch.tensor([seq]))[0, -1].numpy()
        worst = max(worst, float(np.abs(ref - logits[0]).max()))
        nxt = int(rng.choice(len(ref), p=np.exp(ref - ref.max()) / np.exp(ref - ref.max()).sum()))
        seq.append(nxt)
        x = np.array([[nxt]])
    print(f"ONNX check: largest difference from PyTorch {worst:.2e}")
    if worst > 1e-2:
        sys.exit("The exported model doesn't match PyTorch closely enough; please report this.")

    meta = json.load(open(os.path.join(ROOT, "data", args.style, "meta.json"), encoding="utf-8"))
    levels = json.load(open(os.path.join(ROOT, "levels.json"), encoding="utf-8"))
    info = {
        "name": args.run, "label": args.label, "style": meta.get("style", "chopin"),
        "config": cfg.to_dict(), "vocab": tokens.VOCAB, "levels": levels,
        "level_counts": meta["level_counts"], "meters_by_level": meta["meters_by_level"],
        "orig_keys": meta.get("orig_keys", {}),
        "trained_steps": state["step"], "val_loss": state["best"],
        "data": {k: meta[k] for k in ("pieces", "phrases", "sequences", "tokens")},
    }
    with open(os.path.join(exp, "model.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)
    mb = os.path.getsize(onnx_path) / 1e6
    print(f"Exported to {exp} ({mb:.1f} MB).\nNext: python install_model.py --run {args.run}")


if __name__ == "__main__":
    main()
