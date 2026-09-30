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
    return sorted(set(f for f in files if not os.path.basename(f).startswith(".")))

def find_checkpoint(d):
    if os.path.isfile(d): return d
    for c in ["model.pt", "model.pth", "moge.pt"]:
        p = os.path.join(d, c)
        if os.path.isfile(p): return p
    for f in os.listdir(d):
        if f.endswith((".pt", ".pth")): return os.path.join(d, f)
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = find_images()
    ckpt = find_checkpoint(MODEL_DIR)
    print(f"[OK] Checkpoint: {ckpt}")

    from moge.model.v2 import MoGeModel
    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(ckpt).to(device).eval().float()
    print("[OK] MoGe-2 loaded")

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[*] {img_path}")

        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H_img, W_img = img_rgb.shape[:2]
        print(f"    Input image: W={W_img}, H={H_img}")

        t = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

        with torch.no_grad():
            out = model.infer(t, use_fp16=False, resolution_level=5)

        points = out['points'].cpu().numpy()
        mask = out['mask'].cpu().numpy()
        depth = out['depth'].cpu().numpy()
        normal = out['normal'].cpu().numpy() if 'normal' in out else None

        print(f"    MoGe points shape: {points.shape}")
        print(f"    MoGe mask shape: {mask.shape}")
        print(f"    MoGe depth shape: {depth.shape}")

        # Save exactly what MoGe returned — no reshapes
        np.save(os.path.join(OUTPUT_DIR, f"{name}_points_raw.npy"), points)
        np.save(os.path.join(OUTPUT_DIR, f"{name}_mask_raw.npy"), mask)
        np.save(os.path.join(OUTPUT_DIR, f"{name}_depth_raw.npy"), depth)

        # Depth visualization
        dnorm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_depth.png"), (dnorm * 255).astype(np.uint8))

        if normal is not None:
            nv = ((normal + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_normal.png"), nv)

        # PLY for visualization
        H_m, W_m = mask.shape
        pts_flat = points.reshape(-1, 3)
        mask_flat = mask.reshape(-1).astype(bool)
        valid_pts = pts_flat[mask_flat]
        pcd = trimesh.PointCloud(valid_pts)
        pcd.export(os.path.join(OUTPUT_DIR, f"{name}_points.ply"))
        print(f"    Saved: shape={points.shape}, valid={len(valid_pts)}")

if __name__ == "__main__":
    main()
