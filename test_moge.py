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

# ── MAX RESOLUTION ──
RESOLUTION_LEVEL = 9

def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    return sorted(set(f for f in files if not os.path.basename(f).startswith(".")))

def find_checkpoint(d):
    if os.path.isfile(d):
        return d
    for c in ["model.pt", "model.pth", "moge.pt", "checkpoint.pt",
              "pytorch_model.bin", "model.safetensors"]:
        p = os.path.join(d, c)
        if os.path.isfile(p):
            return p
    for f in os.listdir(d):
        if f.endswith((".pt", ".pth", ".safetensors", ".bin")):
            return os.path.join(d, f)
    return None

def safe_normalize(a):
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(a)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = find_images()
    if not images:
        sys.exit(f"[ERROR] No images in {INPUT_DIR}/")

    ckpt = find_checkpoint(MODEL_DIR)
    print(f"[OK] Checkpoint: {ckpt}")

    from moge.model.v2 import MoGeModel
    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(ckpt).to(device).eval().float()
    print("[OK] MoGe-2 loaded on CPU (float32)")

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[*] Processing {img_path}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("    [ERROR] Cannot read image")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_img, W_img = img_rgb.shape[:2]
        print(f"    Input image: W={W_img}, H={H_img}")
        print(f"    Resolution level: {RESOLUTION_LEVEL}")

        t = torch.tensor(
            img_rgb / 255.0, dtype=torch.float32, device=device
        ).permute(2, 0, 1)
        print(f"    Input tensor: {t.shape}")

        with torch.no_grad():
            out = model.infer(t, use_fp16=False, resolution_level=RESOLUTION_LEVEL)

        points = out.get("points")
        mask   = out.get("mask")
        depth  = out.get("depth")
        normal = out.get("normal")

        if points is not None: points = points.cpu().numpy()
        if mask   is not None: mask   = mask.cpu().numpy()
        if depth  is not None: depth  = depth.cpu().numpy()
        if normal is not None: normal = normal.cpu().numpy()

        H_m, W_m = mask.shape
        print(f"    MoGe grid: {W_m}x{H_m}")
        print(f"    Points array: {points.shape}")
        valid_n = int((mask > 0.5).sum())
        print(f"    Valid points: {valid_n}")

        # ── Save full raw grids ──
        np.save(os.path.join(OUTPUT_DIR, f"{name}_points_raw.npy"), points)
        np.save(os.path.join(OUTPUT_DIR, f"{name}_mask_raw.npy"), mask)
        if depth is not None:
            np.save(os.path.join(OUTPUT_DIR, f"{name}_depth_raw.npy"), depth)
        if normal is not None:
            np.save(os.path.join(OUTPUT_DIR, f"{name}_normal_raw.npy"), normal)

        # Metadata
        with open(os.path.join(OUTPUT_DIR, f"{name}_meta.txt"), "w") as f:
            f.write(f"resolution_level={RESOLUTION_LEVEL}\n")
            f.write(f"grid={W_m}x{H_m}\n")
            f.write(f"points={H_m*W_m}\n")
            f.write(f"image={W_img}x{H_img}\n")

        print(f"    [OK] Saved raw grids")

        if depth is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_depth.png"),
                        (safe_normalize(depth) * 255).astype(np.uint8))
            print(f"    [OK] {name}_depth.png")

        if normal is not None:
            normal = np.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)
            nv = ((normal + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_normal.png"), nv)
            print(f"    [OK] {name}_normal.png")

        # PLY preview
        pts_flat = points.reshape(-1, 3)
        m_flat = mask.reshape(-1).astype(bool)
        valid_pts = np.nan_to_num(pts_flat[m_flat], nan=0, posinf=0, neginf=0)
        if len(valid_pts) > 1_000_000:
            valid_pts = valid_pts[np.random.choice(len(valid_pts), 1_000_000, replace=False)]
        trimesh.PointCloud(valid_pts).export(os.path.join(OUTPUT_DIR, f"{name}_points.ply"))
        print(f"    [OK] {name}_points.ply ({len(valid_pts)} pts)")

    print(f"\n[SUCCESS] Outputs saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
