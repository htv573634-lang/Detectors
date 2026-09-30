import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import open3d as o3d

INPUT_IMAGE = "inputs2/test-2.png"
OUTPUT_DIR = "out_moge"

# Fusion controls
DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DA_INTENSITY = 3.0
DA_BLUR_SIGMA = 4.0

def find_latest(pattern):
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0] if files else None

def get_da_depth(image_path):
    from transformers import pipeline
    pipe = pipeline(task="depth-estimation", model=DA_MODEL_ID, device="cpu")
    img = Image.open(image_path).convert("RGB")
    out = pipe(img)
    return np.array(out["depth"], dtype=np.float32)

def build_mesh_from_grid(points, mask):
    """Triangulate the (H,W,3) grid directly — no Poisson needed."""
    H, W = mask.shape
    faces = []
    # Map (i,j) -> vertex index
    idx_map = -np.ones((H, W), dtype=np.int32)
    valid = mask > 0.5
    verts = points[valid]
    idx_map[valid] = np.arange(len(verts))

    for i in range(H - 1):
        for j in range(W - 1):
            a = idx_map[i, j]
            b = idx_map[i + 1, j]
            c = idx_map[i, j + 1]
            d = idx_map[i + 1, j + 1]
            if a >= 0 and b >= 0 and c >= 0:
                faces.append([a, b, c])
            if b >= 0 and d >= 0 and c >= 0:
                faces.append([b, d, c])
    return verts, np.array(faces, dtype=np.int32)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = "test-2"

    # ── Load full MoGe grid ──
    points = np.load(os.path.join(OUTPUT_DIR, f"{name}_points_full.npy"))  # (H,W,3)
    mask   = np.load(os.path.join(OUTPUT_DIR, f"{name}_mask_full.npy"))    # (H,W)
    H_m, W_m = mask.shape
    print(f"[*] MoGe grid: {W_m}x{H_m}")

    moge_depth = points[:, :, 2]  # Z-component = metric depth
    print(f"    MoGe depth range: {moge_depth[mask > 0.5].min():.3f} - {moge_depth[mask > 0.5].max():.3f}")

    # ── Run DA at full MoGe resolution ──
    print("[*] Running Depth Anything V2 Base...")
    da_depth = get_da_depth(INPUT_IMAGE)  # (H_img, W_img)
    da_resized = cv2.resize(da_depth, (W_m, H_m), interpolation=cv2.INTER_LINEAR)
    da_norm = (da_resized - da_resized.min()) / (da_resized.max() - da_resized.min() + 1e-8)

    # ── Regression on valid pixels only ──
    valid = (mask > 0.5) & (np.abs(moge_depth) > 1e-6)
    n_valid = int(valid.sum())
    print(f"    Valid pixels: {n_valid} / {H_m * W_m}")

    X = da_norm[valid]
    Y = moge_depth[valid]
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(f"    Regression: Z ≈ {a:.4f} * da_norm + {b:.4f}   (expected |a| ~ 0.5-2.0)")

    # Auto-flip if regression slope is positive (means DA is inverted)
    if a > 0:
        print(f"    [FIX] DA inverted (a>0). Flipping.")
        da_norm = 1.0 - da_norm
        X = da_norm[valid]
        A = np.vstack([X, np.ones_like(X)]).T
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        print(f"    Corrected: Z ≈ {a:.4f} * da_norm + {b:.4f}")

    da_metric = a * da_norm + b

    # ── High-frequency edge transfer ──
    da_base = cv2.GaussianBlur(da_metric, (0, 0), DA_BLUR_SIGMA)
    da_edges = da_metric - da_base
    print(f"    DA edges: {da_edges[valid].min():.5f} to {da_edges[valid].max():.5f}")

    fused_z = moge_depth + DA_INTENSITY * da_edges
    print(f"    Fused depth range: {fused_z[valid].min():.3f} - {fused_z[valid].max():.3f}")

    # ── Build fused point grid ──
    fused_points = points.copy()
    fused_points[:, :, 2] = fused_z

    # ── Save fused grid ──
    np.save(os.path.join(OUTPUT_DIR, f"{name}_points_fused_full.npy"), fused_points)

    # ── Save vis ──
    vis = fused_z.copy()
    vis[~valid] = vis[valid].min()
    vn = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_depth_fused.png"), (vn * 255).astype(np.uint8))

    edge_vis = ((da_edges - da_edges.min()) / (da_edges.max() - da_edges.min() + 1e-8) * 255).astype(np.uint8)
    edge_vis[~valid] = 0
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_da_edges.png"), edge_vis)

    # ── Direct triangulation (no Poisson needed) ──
    print("[*] Triangulating fused grid...")
    verts, faces = build_mesh_from_grid(fused_points, mask)
    print(f"    {len(verts)} verts, {len(faces)} faces")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # ── Light smoothing to remove grid artifacts ──
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=3)
    o3d_mesh.compute_vertex_normals()

    final = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )

    obj_path = os.path.join(OUTPUT_DIR, f"{name}_mesh_fused.obj")
    glb_path = os.path.join(OUTPUT_DIR, f"{name}_mesh_fused.glb")
    final.export(obj_path)
    final.export(glb_path)
    print(f"[OK] {obj_path}")
    print(f"[OK] {glb_path}")

if __name__ == "__main__":
    main()
