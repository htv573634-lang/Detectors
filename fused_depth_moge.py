import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh

# --- CONFIG ---
INPUT_IMAGE = "inputs2/test-2.png"       # auto-detected below if not found
MOGE_PLY = "out_moge/test-2_points.ply"  # auto-detected below if not found
OUTPUT_DIR = "out_moge"
OUTPUT_PLY = os.path.join(OUTPUT_DIR, "test-2_points_fused.ply")
OUTPUT_FUSED_DEPTH = os.path.join(OUTPUT_DIR, "test-2_depth_fused.png")

# Blend weight: 0.0 = pure DA, 1.0 = pure MoGe
BLEND_MOGE_WEIGHT = 0.6
DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

def find_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file matches: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def find_input_image():
    for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
        for pat in (f"inputs2/*.{ext}", f"inputs2/*.{ext.upper()}"):
            files = glob.glob(pat)
            if files:
                return files[0]
    raise FileNotFoundError("No input image found in inputs2/")

def load_moge_ply(ply_path):
    """Load MoGe PLY, return points as (H*W, 3) array and infer image size."""
    pcd = trimesh.load(ply_path, process=False)
    pts = np.asarray(pcd.vertices)
    # MoGe PLY has points flattened; we need to reshape. MoGe saves points
    # in row-major order of the original image, so we can re-derive grid.
    return pts

def get_da_depth(image_path):
    """Run Depth Anything V2 to get relative depth."""
    print("[*] Loading Depth Anything V2 Small...")
    from transformers import pipeline
    pipe = pipeline(
        task="depth-estimation",
        model=DA_MODEL_ID,
        device="cpu",
    )
    img = Image.open(image_path).convert("RGB")
    out = pipe(img)
    depth = np.array(out["depth"], dtype=np.float32)
    print(f"    DA depth: {depth.shape}, range {depth.min():.3f}-{depth.max():.3f}")
    return depth

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load image ---
    img_path = INPUT_IMAGE if os.path.isfile(INPUT_IMAGE) else find_input_image()
    print(f"[*] Image: {img_path}")
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print(f"    Size: {W}x{H}")

    # --- Load MoGe PLY ---
    ply_path = MOGE_PLY if os.path.isfile(MOGE_PLY) else find_latest("out_moge/*_points.ply")
    print(f"[*] MoGe PLY: {ply_path}")
    pts = load_moge_ply(ply_path)
    print(f"    {len(pts)} points")

    # MoGe's points need to be reshaped to (H_m, W_m, 3).
    # We assume the number of points is a perfect factor of H_m*W_m.
    # The image MoGe used had a certain aspect ratio. We'll try to infer.
    n = len(pts)
    # Try to find H_m, W_m such that H_m*W_m == n and W_m/H_m = W/H
    aspect = W / H
    H_m = int(round(np.sqrt(n / aspect)))
    W_m = int(round(n / H_m))
    while H_m * W_m != n and H_m > 10:
        H_m -= 1
        W_m = int(round(n / H_m))
    if H_m * W_m != n:
        # Fallback: just use square approximation
        H_m = int(round(np.sqrt(n)))
        W_m = int(round(n / H_m))
        if H_m * W_m != n:
            print(f"[ERROR] Cannot reshape {n} points into a grid. Using linear blend.")
            # Fallback to simple nearest-neighbor approach
            return
    print(f"    Reshaped to grid {W_m}x{H_m}")

    points_grid = pts.reshape(H_m, W_m, 3)

    # --- Get MoGe metric depth = Z component of the point grid ---
    moge_depth = points_grid[:, :, 2]
    print(f"    MoGe depth range: {moge_depth.min():.3f} - {moge_depth.max():.3f}")

    # --- Get Depth Anything relative depth, resized to MoGe grid ---
    da_depth = get_da_depth(img_path)
    da_resized = cv2.resize(da_depth, (W_m, H_m), interpolation=cv2.INTER_LINEAR)

    # Normalize DA to [0, 1]
    da_norm = (da_resized - da_resized.min()) / (da_resized.max() - da_resized.min() + 1e-8)

    # --- Align DA to MoGe metric depth via linear regression on valid pixels ---
    # MoGe valid mask = nonzero Z
    valid_mask = np.abs(moge_depth) > 1e-6
    n_valid = int(valid_mask.sum())
    print(f"    Valid pixels: {n_valid} / {H_m * W_m}")

    if n_valid < 100:
        print("[ERROR] Too few valid pixels for regression.")
        return

    X = da_norm[valid_mask]            # (N,)
    Y = moge_depth[valid_mask]         # (N,)
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(f"    Regression: metric_depth ≈ {a:.4f} * da_norm + {b:.4f}")

    # Convert DA to metric depth
    da_metric = a * da_norm + b

    # --- Fuse MoGe metric depth with DA-aligned metric depth ---
    fused_depth = BLEND_MOGE_WEIGHT * moge_depth + (1 - BLEND_MOGE_WEIGHT) * da_metric

    # Preserve invalid regions (background, sky, etc.)
    fused_depth[~valid_mask] = moge_depth[~valid_mask]

    # --- Rebuild point cloud with fused Z ---
    fused_points = points_grid.copy()
    fused_points[:, :, 2] = fused_depth

    # --- Save fused depth visualization ---
    vis = fused_depth.copy()
    vis[~valid_mask] = vis[valid_mask].min()  # fill invalid with min for clean vis
    vis_norm = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    cv2.imwrite(OUTPUT_FUSED_DEPTH, (vis_norm * 255).astype(np.uint8))
    print(f"[OK] Saved fused depth vis: {OUTPUT_FUSED_DEPTH}")

    # --- Save fused PLY ---
    flat_points = fused_points.reshape(-1, 3)
    # Keep only valid points
    valid_flat = valid_mask.reshape(-1)
    flat_points = flat_points[valid_flat]
    print(f"[*] Saving {len(flat_points)} fused points")

    pcd = trimesh.PointCloud(flat_points)
    pcd.export(OUTPUT_PLY)
    print(f"[OK] Saved fused PLY: {OUTPUT_PLY}")

    # Save a backup unfiltered version too (in case)
    pcd_all = trimesh.PointCloud(fused_points.reshape(-1, 3))
    pcd_all.export(OUTPUT_PLY.replace(".ply", "_all.ply"))

    print(f"\n[SUCCESS] Fused point cloud written.")

if __name__ == "__main__":
    main()
