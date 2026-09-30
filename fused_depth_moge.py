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

BLEND_MOGE_WEIGHT = 0.6
DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Poisson reconstruction params
POISSON_DEPTH = 9
DENSITY_QUANTILE = 0.05

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
    print("[*] Loading Depth Anything V2 Small...")
    from transformers import pipeline
    pipe = pipeline(task="depth-estimation", model=DA_MODEL_ID, device="cpu")
    img = Image.open(image_path).convert("RGB")
    out = pipe(img)
    depth = np.array(out["depth"], dtype=np.float32)
    print(f"    DA depth: {depth.shape}, range {depth.min():.3f}-{depth.max():.3f}")
    return depth

def fuse_depths(img_path, ply_path):
    """Fuse MoGe metric depth with DA relative depth. Returns (H,W,3) point grid + mask."""
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print(f"[*] Image size: {W}x{H}")

    pts = load_moge_ply(ply_path)
    n = len(pts)
    print(f"[*] MoGe points: {n}")

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

    da_depth = get_da_depth(img_path)
    da_resized = cv2.resize(da_depth, (W_m, H_m), interpolation=cv2.INTER_LINEAR)
    da_norm = (da_resized - da_resized.min()) / (da_resized.max() - da_resized.min() + 1e-8)

    valid_mask = np.abs(moge_depth) > 1e-6
    n_valid = int(valid_mask.sum())
    print(f"    Valid pixels: {n_valid} / {H_m * W_m}")

    if n_valid < 100:
        raise RuntimeError("Too few valid pixels for regression")

    X = da_norm[valid_mask]
    Y = moge_depth[valid_mask]
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(f"    Regression: metric ≈ {a:.4f} * da_norm + {b:.4f}")

    da_metric = a * da_norm + b
    fused_depth = BLEND_MOGE_WEIGHT * moge_depth + (1 - BLEND_MOGE_WEIGHT) * da_metric
    fused_depth[~valid_mask] = moge_depth[~valid_mask]

    fused_points = points_grid.copy()
    fused_points[:, :, 2] = fused_depth

    # Vis
    vis = fused_depth.copy()
    vis[~valid_mask] = vis[valid_mask].min()
    vis_norm = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    cv2.imwrite(OUTPUT_FUSED_DEPTH, (vis_norm * 255).astype(np.uint8))
    print(f"[OK] Fused depth vis: {OUTPUT_FUSED_DEPTH}")

    return fused_points, valid_mask

def poisson_mesh_from_points(pts, output_obj):
    """Reconstruct mesh from point cloud using Open3D Poisson."""
    print(f"[*] Poisson reconstruction from {len(pts)} points...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if len(pcd.points) > 100000:
        print(f"    Downsampling to 100k...")
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

    print("[*] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

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
    print(f"[*] Image: {img_path}")

    ply_path = MOGE_PLY if os.path.isfile(MOGE_PLY) else find_latest("out_moge/*_points.ply")
    print(f"[*] MoGe PLY: {ply_path}")

    # Fuse
    fused_points, valid_mask = fuse_depths(img_path, ply_path)

    # Save fused PLY
    flat = fused_points.reshape(-1, 3)
    valid_flat = valid_mask.reshape(-1)
    flat_valid = flat[valid_flat]
    print(f"[*] Saving {len(flat_valid)} valid fused points")

    pcd = trimesh.PointCloud(flat_valid)
    pcd.export(OUTPUT_PLY)
    print(f"[OK] PLY: {OUTPUT_PLY}")

    pcd_all = trimesh.PointCloud(flat)
    pcd_all.export(OUTPUT_PLY.replace(".ply", "_all.ply"))

    # Mesh from fused points
    print("\n[*] Converting fused PLY to OBJ mesh...")
    poisson_mesh_from_points(flat_valid, OUTPUT_OBJ)

    print("\n[SUCCESS] Done.")

if __name__ == "__main__":
    main()
