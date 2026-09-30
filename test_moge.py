import os
import glob
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh

sys.path.insert(0, "MoGe")

INPUT_DIR = "inputs2"
OUTPUT_DIR = "out_moge"
MODEL_DIR = "MoGe/checkpoints/moge-2-vitl-normal"

IMAGE_EXTS = ("jpg", "jpeg", "jpge", "png", "bmp", "webp")

def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
    return files

def find_checkpoint(model_dir):
    if os.path.isfile(model_dir):
        return model_dir
    if not os.path.isdir(model_dir):
        return None
    for c in ["model.pt", "model.pth", "moge.pt", "checkpoint.pt",
              "pytorch_model.bin", "model.safetensors"]:
        p = os.path.join(model_dir, c)
        if os.path.isfile(p):
            return p
    for f in os.listdir(model_dir):
        if f.endswith((".pt", ".pth", ".safetensors", ".bin")):
            return os.path.join(model_dir, f)
    return None

def safe_normalize(arr):
    """Normalize to [0,1] with NaN/inf protection."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-6:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = find_images()
    if not images:
        print(f"[ERROR] No images found in {INPUT_DIR}/")
        sys.exit(1)
    print(f"[*] Found {len(images)} image(s)")

    ckpt_file = find_checkpoint(MODEL_DIR)
    if ckpt_file is None:
        print(f"[ERROR] No checkpoint file found in {MODEL_DIR}")
        sys.exit(1)
    print(f"[OK] Using checkpoint: {ckpt_file}")

    from moge.model.v2 import MoGeModel
    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(ckpt_file).to(device).eval()
    model = model.float()  # Force float32 for CPU
    print("[OK] MoGe-2 loaded on CPU (float32)")

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[*] Processing {img_path}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"    [ERROR] Could not read image")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_tensor = torch.tensor(
            img_rgb / 255.0, dtype=torch.float32, device=device
        ).permute(2, 0, 1)
        print(f"    Input tensor: shape={img_tensor.shape}, dtype={img_tensor.dtype}")

        with torch.no_grad():
            output = model.infer(img_tensor, use_fp16=False, resolution_level=5)

        points = output.get('points')
        depth = output.get('depth')
        mask = output.get('mask')
        normal = output.get('normal')

        if points is not None: points = points.cpu().numpy()
        if depth is not None: depth = depth.cpu().numpy()
        if mask is not None: mask = mask.cpu().numpy()
        if normal is not None: normal = normal.cpu().numpy()

        # ── Depth map with NaN protection ──
        if depth is not None:
            dv = safe_normalize(depth)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"{name}_depth.png"),
                (dv * 255).astype(np.uint8),
            )
            print(f"    [OK] {name}_depth.png")

        # ── Normal map ──
        if normal is not None:
            normal = np.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)
            nv = ((normal + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_normal.png"), nv)
            print(f"    [OK] {name}_normal.png")

        # ── Point cloud ──
        if points is not None and mask is not None:
            pts = points.reshape(-1, 3)
            m = mask.reshape(-1).astype(bool)
            pts = pts[m]
            pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
            if len(pts) > 0:
                if len(pts) > 50000:
                    idx = np.random.choice(len(pts), 50000, replace=False)
                    pts = pts[idx]
                pcd = trimesh.PointCloud(pts)
                pcd.export(os.path.join(OUTPUT_DIR, f"{name}_points.ply"))
                print(f"    [OK] {name}_points.ply ({len(pts)} pts)")

    print(f"\n[SUCCESS] Outputs saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
