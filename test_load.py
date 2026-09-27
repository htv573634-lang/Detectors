import os
import sys
import gc
import torch
import subprocess

print("=== PyTorch version:", torch.__version__)
print("=== Before load ===")
subprocess.run(["free", "-h"])

ckpt_path = "ckpt/checkpoint.ckpt"
if not os.path.exists(ckpt_path):
    print(f"[FAIL] Missing: {ckpt_path}")
    sys.exit(1)

size = os.path.getsize(ckpt_path)
print(f"[INFO] Loading: {ckpt_path} ({size/1024/1024:.1f} MB)")

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

try:
    data = torch.load(ckpt_path, map_location="cpu")
    print("[OK] Checkpoint loaded")
    print(f"[INFO] Top-level keys: {list(data.keys())[:10]}")
    if "state_dict" in data:
        print(f"[INFO] state_dict entries: {len(data['state_dict'])}")
    print("=== After load ===")
    subprocess.run(["free", "-h"])
    del data
    gc.collect()
    print("=== After cleanup ===")
    subprocess.run(["free", "-h"])
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    subprocess.run(["free", "-h"])
    raise
