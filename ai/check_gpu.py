"""Check that PyTorch can train on your graphics card before downloading or training anything.

  python check_gpu.py
"""
import sys
import time

try:
    import torch
except ImportError:
    sys.exit("PyTorch isn't installed. Follow step 3 in README.md.")

print(f"PyTorch {torch.__version__}")
hip = getattr(torch.version, "hip", None)
print(f"ROCm/HIP: {hip or 'not in this build'}")
if not torch.cuda.is_available():
    print("\nNo usable GPU found. Training would run on the CPU (very slow).")
    print("Check: AMD driver version (README step 1), and that you installed AMD's ROCm build of PyTorch (step 3).")
    sys.exit(1)

dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name(0)}  ({props.total_memory / 2**30:.1f} GB)")

# 1. raw speed
a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
torch.cuda.synchronize(); t = time.time()
for _ in range(50):
    a = a @ a.T / 2048
torch.cuda.synchronize()
print(f"Matrix multiply: {50 * 2 * 2048**3 / (time.time() - t) / 1e12:.1f} TFLOPS (fp16)")

# 2. a tiny training run with the same code path as train.py
from fsai.model import Config, MusicGPT
for prec, sdpa in (("fp16", True), ("fp16", False), ("fp32", True)):
    try:
        torch.manual_seed(0)
        m = MusicGPT(Config(vocab_size=230, max_len=256, n_layer=2, n_head=4, d_model=128), sdpa=sdpa).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=prec == "fp16")
        x = torch.randint(3, 230, (32, 200), device=dev)
        t = time.time()
        for i in range(30):
            with torch.autocast("cuda", dtype=torch.float16, enabled=prec == "fp16"):
                _, loss = m(x[:, :-1], x[:, 1:])
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        ok = torch.isfinite(loss).item()
        print(f"Training test ({prec}, {'fused' if sdpa else 'plain'} attention): {'OK' if ok else 'NaN!'}  {30 / (time.time() - t):.1f} steps/s")
        if ok and prec == "fp16" and sdpa:
            print("\nEverything works. Next: python get_data.py")
            break
        if ok and not sdpa:
            print("\nWorks with plain attention: train with  python train.py --no-sdpa")
            break
        if ok and prec == "fp32":
            print("\nWorks in full precision: train with  python train.py --precision fp32")
            break
    except Exception as e:
        print(f"Training test ({prec}, {'fused' if sdpa else 'plain'} attention) failed: {e}")
else:
    sys.exit("\nTraining doesn't work on this GPU yet. See Troubleshooting in README.md.")
