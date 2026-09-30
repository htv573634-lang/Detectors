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

# Intensity controls — used ONLY if correlation proves meaningful
DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DA_INTENSITY = 2.0
DA_BLUR_SIGMA = 4.0
CORRELATION_THRESHOLD = 0.30   # below this, DA is ignored entirely

def get_da_depth(image_path):
    from transformers import pipeline
    pipe = pipeline(task="depth-estimation", model=DA_MODEL_ID, device="cpu")
    img = Image.open(image_path).convert("RGB")
    out = pipe(img)
    return np.array(out["depth"], dtype=np.float32)

def triangulate(points, mask):
    """Build mesh from MoGe's native grid without resampling."""
    H, W = mask.shape
    verts = []
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
    print(f"[*] MoGe grid: H_m={H_m}, W_m={W_m}")
    print(f"    MoGe depth range: {moge_depth[mask > 0.5].min():.3f} - {moge_depth[mask > 0.5].max():.3f}")

    # ── Load DA at native image resolution ──
    print("[*] Running Depth Anything V2 Base...")
    da_depth = get_da_depth(INPUT_IMAGE)
    print(f"    DA shape: {da_depth.shape}")

    # ── Resize DA to MoGe's grid (H_m, W_m) with correct cv2 order ──
    # cv2.resize takes (width, height) — so pass (W_m, H_m) to get shape (H_m, W_m)
    da_resized = cv2.resize(da_depth, (W_m, H_m), interpolation=cv2.INTER_LINEAR)
    print(f"    DA resized to: {da_resized.shape}")

    # ── Normalize DA to [0, 1] ──
    da_norm = (da_resized - da_resized.min()) / (da_resized.max() - da_resized.min() + 1e-8)

    # ── Regression on valid pixels only ──
    valid = (mask > 0.5) & (np.abs(moge_depth) > 1e-6)
    n_valid = int(valid.sum())
    print(f"    Valid pixels: {n_valid} / {H_m * W_m}")

    X = da_norm[valid]
    Y = moge_depth[valid]

    corr = np.corrcoef(X, Y)[0, 1]
    print(f"    Correlation(DA, MoGe_depth) = {corr:.4f}")

    # ── Decision: use DA only if correlation is meaningful ──
    if abs(corr) < CORRELATION_THRESHOLD:
        print(f"    [!] Correlation below threshold ({CORRELATION_THRESHOLD}).")
        print(f"    [!] DA cannot refine MoGe depth for this image. Using MoGe alone.")
        fused_depth = moge_depth.copy()
    else:
        # Fit linear regression
        A = np.vstack([X, np.ones_like(X)]).T
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        print(f"    Regression: Z ≈ {a:.4f} * DA + {b:.4f}")

        # Auto-flip if positive (means DA is inverted relative to MoGe)
        if a > 0:
            print(f"    [FIX] DA inverted. Flipping.")
            da_norm = 1.0 - da_norm
            X = da_norm[valid]
            A = np.vstack([X, np.ones_like(X)]).T
            a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
            print(f"    Corrected: Z ≈ {a:.4f} * DA + {b:.4f}")

        da_metric = a * da_norm + b
        da_base = cv2.GaussianBlur(da_metric, (0, 0), DA_BLUR_SIGMA)
        da_edges = da_metric - da_base
        print(f"    DA edges std: {da_edges[valid].std():.5f}")

        # Only apply if edges are meaningful
        if da_edges[valid].std() < 1e-4:
            print(f"    [!] DA edges are negligible. Using MoGe alone.")
            fused_depth = moge_depth.copy()
        else:
            print(f"    Applying DA edges (intensity={DA_INTENSITY})...")
            fused_depth = moge_depth + DA_INTENSITY * da_edges
            fused_depth[~valid] = moge_depth[~valid]

    # ── Build mesh from MoGe's native grid ──
    print("[*] Triangulating MoGe grid...")
    fused_points = points.copy()
    # MoGe returns points as (H_m, W_m, 3); we replace Z with fused_depth
    fused_points[:, :, 2] = fused_depth

    verts, faces = triangulate(fused_points, mask)
    print(f"    {len(verts)} verts, {len(faces)} faces")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Light smoothing
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_vertices()
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
