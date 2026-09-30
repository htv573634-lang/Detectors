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
    """Find the actual .pt/.pth/.safetensors file inside the directory."""
    if os.path.isfile(model_dir):
        return model_dir
    if not os.path.isdir(model_dir):
        return None
    # Preferred names
    for c in ["model.pt", "model.pth", "moge.pt", "checkpoint.pt", "pytorch_model.bin", "model.safetensors"]:
        p = os.path.join(model_dir, c)
        if os.path.isfile(p):
            return p
    # Fallback: any model-like file
    for f in os.listdir(model_dir):
        if f.endswith((".pt", ".pth", ".safetensors", ".bin")):
            return os.path.join(model_dir, f)
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = find_images()
    if not images:
        print(f"[ERROR] No images found in {INPUT_DIR}/")
        sys.exit(1)
    print(f"[*] Found {len(images)} image(s)")

    print(f"[*] Looking for checkpoint in: {MODEL_DIR}")
    ckpt_file = find_checkpoint(MODEL_DIR)
    if ckpt_file is None:
        print(f"[ERROR] No checkpoint file found in {MODEL_DIR}")
        if os.path.isdir(MODEL_DIR):
            print(f"        Directory contents: {os.listdir(MODEL_DIR)}")
        sys.exit(1)
    print(f"[OK] Using checkpoint: {ckpt_file}")

    from moge.model.v2 import MoGeModel
    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(ckpt_file).to(device).eval()
    print("[OK] MoGe-2 loaded on CPU")

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[*] Processing {img_path}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"    [ERROR] Could not read image")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1] and convert to tensor (3, H, W)
        img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

        with torch.no_grad():
            output = model.infer(img_tensor)

        # Extract
        points = output.get('points')
        depth = output.get('depth')
        mask = output.get('mask')
        normal = output.get('normal')

        if points is not None: points = points.cpu().numpy()
        if depth is not None: depth = depth.cpu().numpy()
        if mask is not None: mask = mask.cpu().numpy()
        if normal is not None: normal = normal.cpu().numpy()

        # Depth vis
        if depth is not None:
            dv = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_depth.png"), (dv * 255).astype(np.uint8))
            print(f"    [OK] {name}_depth.png")

        # Normal vis
        if normal is not None:
            nv = ((normal + 1.0) / 2.0 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_normal.png"), nv)
            print(f"    [OK] {name}_normal.png")

        # Point cloud
        if points is not None and mask is not None:
            pts = points.reshape(-1, 3)
            m = mask.reshape(-1).astype(bool)
            pts = pts[m]
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
