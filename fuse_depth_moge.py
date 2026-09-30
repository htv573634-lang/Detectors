import os
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import open3d as o3d

INPUT_IMAGE = "inputs2/test-2.png"
OUTPUT_DIR = "out_moge"

# ── DA SETTINGS ──
DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DA_INTENSITY = 0.50
MAX_DELTA_METERS = 0.008
DA_BLUR_SIGMA = 6.0
CORRELATION_THRESHOLD = 0.30

# ── LINE-REMOVAL SETTINGS ──
DEPTH_BILATERAL_D = 7          # filter diameter
DEPTH_BILATERAL_SIGMA_COLOR = 0.02   # ← LOW value preserves features
DEPTH_BILATERAL_SIGMA_SPACE = 4      # ← only smooths close neighbors (row-to-row)
DEPTH_GAUSSIAN_SIGMA = 0.8     # gentle Gaussian after bilateral

# ── MESH QUALITY ──
SUBDIVIDE_PASSES = 2           # was 1 → now 2 (finer surface)
SMOOTH_ITERS = 8               # was 10 → now 8 (preserve detail on finer mesh)


def find_name():
    metas = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_meta.txt")]
    if not metas:
        return "test-2"
    metas.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
    return metas[0].replace("_meta.txt", "")


def get_da_depth(image_path):
    from transformers import pipeline
    pipe = pipeline(task="depth-estimation", model=DA_MODEL_ID, device="cpu")
    img = Image.open(image_path).convert("RGB")
    out = pipe(img)
    return np.array(out["depth"], dtype=np.float32)


def triangulate(points, mask):
    H, W = mask.shape
    idx = -np.ones((H, W), dtype=np.int32)
    v = mask > 0.5
    verts = points[v]
    idx[v] = np.arange(len(verts))
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = idx[i, j]; b = idx[i+1, j]; c = idx[i, j+1]; d = idx[i+1, j+1]
            if a >= 0 and b >= 0 and c >= 0:
                faces.append([a, b, c])
            if b >= 0 and d >= 0 and c >= 0:
                faces.append([b, d, c])
    return verts, np.array(faces, dtype=np.int32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    NAME = find_name()
    print(f"[*] Processing: {NAME}")

    # Load MoGe raw grids
    points = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_points_raw.npy"))
    mask   = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_mask_raw.npy"))
    moge_depth = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_depth_raw.npy"))
    H_m, W_m = mask.shape
    print(f"[*] MoGe grid: {W_m}x{H_m}")

    valid = (mask > 0.5) & (np.abs(moge_depth) > 1e-6)
    print(f"    Valid pixels: {int(valid.sum())}")
    print(f"    MoGe depth range: {moge_depth[valid].min():.3f} - {moge_depth[valid].max():.3f}")

    # ── DA inference ──
    print(f"[*] Loading DA model: {DA_MODEL_ID}")
    da_depth = get_da_depth(INPUT_IMAGE)
    da_resized = cv2.resize(da_depth, (W_m, H_m), interpolation=cv2.INTER_LINEAR)
    da_norm = (da_resized - da_resized.min()) / (da_resized.max() - da_resized.min() + 1e-8)

    # ── Correlation ──
    X = da_norm[valid]
    Y = moge_depth[valid]
    corr = np.corrcoef(X, Y)[0, 1]
    print(f"\n===== CORRELATION ===== ")
    print(f"    corr(DA, MoGe_depth) = {corr:.4f}")

    delta_z = np.zeros_like(moge_depth)

    if abs(corr) >= CORRELATION_THRESHOLD:
        A = np.vstack([X, np.ones_like(X)]).T
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        if a > 0:
            print(f"    [FIX] DA inverted. Flipping.")
            da_norm = 1.0 - da_norm
            X = da_norm[valid]
            A = np.vstack([X, np.ones_like(X)]).T
            a, b = np.linalg.lstsq(A, Y, rcond=None)[0]

        da_metric = a * da_norm + b
        da_base = cv2.GaussianBlur(da_metric, (0, 0), DA_BLUR_SIGMA)
        da_edges = da_metric - da_base

        raw_delta = DA_INTENSITY * da_edges
        delta_z = np.clip(raw_delta, -MAX_DELTA_METERS, MAX_DELTA_METERS)
        print(f"    Delta range = {delta_z[valid].min():.5f} to {delta_z[valid].max():.5f}")
    else:
        print(f"    [SKIP] Correlation too low")

    # ── Apply DA delta ──
    fused_depth = moge_depth + delta_z
    fused_points = points.copy()
    fused_points[:, :, 2] = fused_depth

    # ══════════════════════════════════════════════
    # ── LINE-REMOVAL STEP: SMOOTH THE DEPTH GRID ──
    # Bilateral filter removes row-to-row jumps.
    # Applied to the Z-channel of the point grid.
    # ══════════════════════════════════════════════
    print("\n[*] Applying bilateral filter to remove grid terracing...")

    # First: bilateral on the depth (row-to-row smoothing, feature-preserving)
    z_channel = fused_points[:, :, 2].astype(np.float32)
    z_smooth = cv2.bilateralFilter(
        z_channel,
        d=DEPTH_BILATERAL_D,
        sigmaColor=DEPTH_BILATERAL_SIGMA_COLOR,
        sigmaSpace=DEPTH_BILATERAL_SIGMA_SPACE,
    )

    # Second: gentle Gaussian to blend the seams
    z_smooth = cv2.GaussianBlur(z_smooth, (0, 0), DEPTH_GAUSSIAN_SIGMA)

    # Preserve mask: don't smooth over invalid pixels
    z_smooth[~valid] = z_channel[~valid]

    fused_points[:, :, 2] = z_smooth
    print(f"    Z-channel: raw range [{z_channel[valid].min():.4f}, {z_channel[valid].max():.4f}]")
    print(f"    Z-channel: smoothed range [{z_smooth[valid].min():.4f}, {z_smooth[valid].max():.4f}]")

    # ── Triangulate ──
    print("\n[*] Triangulating...")
    verts, faces = triangulate(fused_points, mask)
    print(f"    Raw: {len(verts)} verts, {len(faces)} faces")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_unreferenced_vertices()

    # ── Subdivision (2 passes for finer surface) ──
    for i in range(SUBDIVIDE_PASSES):
        print(f"[*] Subdivision pass {i+1}/{SUBDIVIDE_PASSES}...")
        o3d_mesh = o3d_mesh.subdivide_midpoint(number_of_iterations=1)
        print(f"    {len(o3d_mesh.vertices)} verts, {len(o3d_mesh.triangles)} faces")

    # ── Taubin smoothing ──
    print(f"[*] Taubin smoothing ({SMOOTH_ITERS} iterations)...")
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERS)
    o3d_mesh.compute_vertex_normals()

    final = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )

    obj_path = os.path.join(OUTPUT_DIR, f"{NAME}_mesh_fused.obj")
    glb_path = os.path.join(OUTPUT_DIR, f"{NAME}_mesh_fused.glb")
    final.export(obj_path)
    final.export(glb_path)
    print(f"\n[OK] {obj_path}")
    print(f"[OK] {glb_path}")
    print(f"    GLB size: {os.path.getsize(glb_path)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
