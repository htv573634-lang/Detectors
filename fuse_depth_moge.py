import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import open3d as o3d

# --- CONFIG ---
INPUT_IMAGE = "inputs2/test-2.png"
MOGE_PLY = "out_moge/test-2_points.ply"
OUTPUT_DIR = "out_moge"
OUTPUT_PLY = os.path.join(OUTPUT_DIR, "test-2_points_fused.ply")
OUTPUT_OBJ = os.path.join(OUTPUT_DIR, "test-2_mesh_fused.obj")
OUTPUT_GLB = os.path.join(OUTPUT_DIR, "test-2_mesh_fused.glb")
OUTPUT_FUSED_DEPTH = os.path.join(OUTPUT_DIR, "test-2_depth_fused.png")

# --- INTENSITY CONTROLS ---
DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DA_INTENSITY = 4.0           # ⭐ THE MAIN KNOB: 1.0=subtle, 4.0=strong, 6.0=extreme
DA_BLUR_SIGMA = 5.0          # Higher = only very sharp edges pass through
MOGE_SMOOTH_SIGMA = 0.0      # Set >0 to smooth MoGe before adding DA edges

POISSON_DEPTH = 11
DENSITY_QUANTILE = 0.02

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
    pcd = trimesh.load(ply_path, process=False)
    return np.asarray(pcd.vertices)

def get_da_depth(image_path):
    print(f"[*] Loading Depth Anything V2 Base ({DA_MODEL_ID})...")
    from transformers import pipeline
    pipe = pipeline(task="depth-estimation", model=DA_MODEL_ID, device="cpu")
    img = Image.open(image_path).convert("RGB")
    out = pipe(img)
    depth = np.array(out["depth"], dtype=np.float32)
    print(f"    DA depth: {depth.shape}, range {depth.min():.3f}-{depth.max():.3f}")
    return depth

def fuse_depths(img_path, ply_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print(f"[*] Image size: {W}x{H}")

    pts = load_moge_ply(ply_path)
    n = len(pts)
    print(f"[*] MoGe points: {n}")

    # Determine grid shape
    aspect = W / H
    H_m = int(round(np.sqrt(n / aspect)))
    W_m = int(round(n / H_m))
    while H_m * W_m != n and H_m > 10:
        H_m -= 1
        W_m = int(round(n / H_m))
    if H_m * W_m != n:
        H_m = int(round(np.sqrt(n)))
        W_m = int(round(n / H_m))
        if H_m * W_m != n:
            raise RuntimeError(f"Cannot reshape {n} points into a grid")
    print(f"    Grid: {W_m}x{H_m}")

    points_grid = pts.reshape(H_m, W_m, 3)
    moge_depth = points_grid[:, :, 2]
    print(f"    MoGe depth range: {moge_depth.min():.3f} - {moge_depth.max():.3f}")

    # --- Get DA and resize ---
    da_depth = get_da_depth(img_path)
    da_resized = cv2.resize(da_depth, (W_m, H_m), interpolation=cv2.INTER_LINEAR)
    da_norm = (da_resized - da_resized.min()) / (da_resized.max() - da_resized.min() + 1e-8)

    # --- Align DA to MoGe metric via regression ---
    valid_mask = np.abs(moge_depth) > 1e-6
    n_valid = int(valid_mask.sum())
    print(f"    Valid pixels: {n_valid} / {H_m * W_m}")

    X = da_norm[valid_mask]
    Y = moge_depth[valid_mask]
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(f"    Regression: metric ≈ {a:.4f} * da_norm + {b:.4f}")

    da_metric = a * da_norm + b

    # --- HIGH-FREQUENCY TRANSFER ---
    # Step 1: smooth DA → get base layer
    da_base = cv2.GaussianBlur(da_metric, (0, 0), DA_BLUR_SIGMA)
    # Step 2: extract edges/details = da - da_base
    da_edges = da_metric - da_base
    # Step 3: optionally smooth MoGe
    if MOGE_SMOOTH_SIGMA > 0:
        moge_base = cv2.GaussianBlur(moge_depth, (0, 0), MOGE_SMOOTH_SIGMA)
    else:
        moge_base = moge_depth.copy()
    # Step 4: fused = MoGe base + amplified DA edges
    fused_depth = moge_base + DA_INTENSITY * da_edges

    print(f"[*] DA edges range: {da_edges.min():.4f} - {da_edges.max():.4f}")
    print(f"[*] DA_INTENSITY = {DA_INTENSITY}")
    print(f"[*] Fused depth range: {fused_depth.min():.3f} - {fused_depth.max():.3f}")

    # Preserve invalid regions
    fused_depth[~valid_mask] = moge_depth[~valid_mask]

    fused_points = points_grid.copy()
    fused_points[:, :, 2] = fused_depth

    # Save vis
    vis = fused_depth.copy()
    vis[~valid_mask] = vis[valid_mask].min()
    vis_norm = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    cv2.imwrite(OUTPUT_FUSED_DEPTH, (vis_norm * 255).astype(np.uint8))
    print(f"[OK] Fused depth vis: {OUTPUT_FUSED_DEPTH}")

    # Also save the edges map alone for debugging
    edges_vis = ((da_edges - da_edges.min()) / (da_edges.max() - da_edges.min() + 1e-8) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "test-2_da_edges.png"), edges_vis)

    return fused_points, valid_mask

def poisson_mesh_from_points(pts, output_obj):
    print(f"[*] Poisson reconstruction from {len(pts)} points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    print("[*] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=40)
    )
    pcd.orient_normals_consistent_tangent_plane(40)

    print(f"[*] Poisson (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )

    densities = np.asarray(densities)
    to_remove = densities < np.quantile(densities, DENSITY_QUANTILE)
    mesh.remove_vertices_by_mask(to_remove)
    mesh.compute_vertex_normals()

    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )
    tm.export(output_obj)
    print(f"[OK] OBJ: {output_obj} ({len(tm.vertices)} verts, {len(tm.faces)} faces)")

    glb_path = output_obj.replace(".obj", ".glb")
    tm.export(glb_path)
    print(f"[OK] GLB: {glb_path}")
    return tm

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_path = INPUT_IMAGE if os.path.isfile(INPUT_IMAGE) else find_input_image()
    ply_path = MOGE_PLY if os.path.isfile(MOGE_PLY) else find_latest("out_moge/*_points.ply")

    fused_points, valid_mask = fuse_depths(img_path, ply_path)

    flat = fused_points.reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)
    flat_valid = flat[valid_flat]
    print(f"[*] Saving {len(flat_valid)} valid fused points")

    pcd = trimesh.PointCloud(flat_valid)
    pcd.export(OUTPUT_PLY)
    print(f"[OK] PLY: {OUTPUT_PLY}")

    print("\n[*] Converting fused PLY to OBJ mesh...")
    poisson_mesh_from_points(flat_valid, OUTPUT_OBJ)
    print("\n[SUCCESS] Done.")

if __name__ == "__main__":
    main()
