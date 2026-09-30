import os
import glob
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh

# Add MoGe to path
sys.path.insert(0, "MoGe")

INPUT_DIR = "inputs2"
OUTPUT_DIR = "out_moge"
MODEL_PATH = "MoGe/checkpoints/moge-2-vitl-normal"

IMAGE_EXTS = ("jpg", "jpeg", "jpge", "png", "bmp", "webp")

def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
    return files

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = find_images()
    if not images:
        print(f"[ERROR] No images found in {INPUT_DIR}/")
        sys.exit(1)
    print(f"[*] Found {len(images)} image(s)")

    # Load MoGe-2 model
    print(f"[*] Loading MoGe-2 from {MODEL_PATH}...")
    from moge.model.v2 import MoGeModel
    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(MODEL_PATH).to(device).eval()
    print("[OK] MoGe-2 loaded on CPU")

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[*] Processing {img_path}")

        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"    [ERROR] Could not read image")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Convert to tensor (MoGe expects (3, H, W) normalized to [0, 1])
        img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

        # Run inference
        with torch.no_grad():
            output = model.infer(img_tensor)

        # Extract outputs
        points = output['points'].cpu().numpy()      # (H, W, 3)
        depth = output['depth'].cpu().numpy()        # (H, W)
        mask = output['mask'].cpu().numpy()          # (H, W)
        normal = output['normal'].cpu().numpy() if 'normal' in output else None

        # --- Save Depth Map ---
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_depth.png"), depth_vis)
        print(f"    [OK] Saved {name}_depth.png")

        # --- Save Normal Map ---
        if normal is not None:
            # Normal map is in [-1, 1]; convert to [0, 255]
            normal_vis = ((normal + 1.0) / 2.0 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_normal.png"), normal_vis)
            print(f"    [OK] Saved {name}_normal.png")
        else:
            print(f"    [WARN] No normal map in output")

        # --- Save 3D Point Cloud (PLY) ---
        if points is not None and mask is not None:
            # Flatten and filter by mask
            pts = points.reshape(-1, 3)
            m = mask.reshape(-1).astype(bool)
            pts = pts[m]

            if len(pts) > 0:
                # Downsample for speed
                if len(pts) > 50000:
                    idx = np.random.choice(len(pts), 50000, replace=False)
                    pts = pts[idx]

                pcd = trimesh.PointCloud(pts)
                ply_path = os.path.join(OUTPUT_DIR, f"{name}_points.ply")
                pcd.export(ply_path)
                print(f"    [OK] Saved {name}_points.ply ({len(pts)} points)")

    print(f"\n[SUCCESS] All outputs saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
