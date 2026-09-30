import os
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import open3d as o3d

INPUT_IMAGE = "inputs2/test-2.png"
OUTPUT_DIR = "out_moge"
NAME = "test-2"

DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"

# ── CALIBRATED CONTROLS ──
DA_INTENSITY = 0.5           # Keep low: 0.3 = subtle, 1.0 = strong
MAX_DELTA_METERS = 0.008     # HARD CAP: ±8 mm per pixel
DA_BLUR_SIGMA = 6.0
CORRELATION_THRESHOLD = 0.30
SMOOTH_ITERATIONS = 10       # Taubin smoothing iterations
SUBDIVIDE = True             # Midpoint subdivision to smooth grid steps

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

    # ── Load MoGe's native output ──
    points = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_points_raw.npy"))
    mask = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_mask_raw.npy"))
    moge_depth = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_depth_raw.npy"))
    H_m, W_m = mask.shape
    print(f"[*] MoGe grid: {W_m}x{H_m}")
    print(f"    MoGe depth range: {moge_depth[mask > 0.5].min():.3f} - {moge_depth[mask > 0.5].max():.3f}")

    # ── DA inference ──
    print("[*] Running Depth Anything V2 Base...")
    da_depth = get_da_depth(INPUT_IMAGE)
    da_resized = cv2.resize(da_depth, (W_m, H_m), interpolation=cv2.INTER_LINEAR)
    da_norm = (da_resized - da_resized.min()) / (da_resized.max() - da_resized.min() + 1e-8)

    # ── Regression ──
    valid = (mask > 0.5) & (np.abs(moge_depth) > 1e-6)
    X = da_norm[valid]
    Y = moge_depth[valid]
    corr = np.corrcoef(X, Y)[0, 1]
    print(f"    Correlation = {corr:.4f}")

    delta_z = np.zeros_like(moge_depth)

    if abs(corr) >= CORRELATION_THRESHOLD:
        A = np.vstack([X, np.ones_like(X)]).T
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        if a > 0:
            da_norm = 1.0 - da_norm
            X = da_norm[valid]
            A = np.vstack([X, np.ones_like(X)]).T
            a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        da_metric = a * da_norm + b
        da_base = cv2.GaussianBlur(da_metric, (0, 0), DA_BLUR_SIGMA)
        da_edges = da_metric - da_base

        raw_delta = DA_INTENSITY * da_edges
        delta_z = np.clip(raw_delta, -MAX_DELTA_METERS, MAX_DELTA_METERS)
        print(f"    DA edges applied (clamped to ±{MAX_DELTA_METERS} m)")
    else:
        print(f"    [!] Correlation too low. Using MoGe alone.")

    # ── Apply delta as ADDITION ──
    fused_points = points.copy()
    fused_points[:, :, 2] += delta_z

    # ── Triangulate ──
    print("[*] Triangulating...")
    verts, faces = triangulate(fused_points, mask)
    print(f"    Raw: {len(verts)} verts, {len(faces)} faces")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # ── CALIBRATION: Subdivision + Taubin Smoothing ──
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_vertices()

    if SUBDIVIDE:
        print("[*] Subdividing to smooth grid steps...")
        o3d_mesh = o3d_mesh.subdivide_midpoint(number_of_iterations=1)
        print(f"    After subdivision: {len(o3d_mesh.vertices)} verts")

    print(f"[*] Applying Taubin smoothing ({SMOOTH_ITERATIONS} iterations)...")
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERATIONS)
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
    print(f"[OK] {obj_path}")
    print(f"[OK] {glb_path}")

if __name__ == "__main__":
    main()
