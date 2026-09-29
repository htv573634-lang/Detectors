import os
import numpy as np
import cv2
import trimesh
from PIL import Image
from transformers import pipeline

# --- CONFIGURATION ---
INPUT_MESH = "out_sam/test-2_mesh.obj"
INPUT_IMAGE = "inputs/test-2.png"
OUTPUT_MESH = "out_sam/test-2_mesh_refined.obj"
OUTPUT_DEPTH_VIS = "out_sam/test-2_depth_vis.png"

FOCAL_LENGTH = 718.9
CX, CY = 256.0, 256.0
CAM_T = np.array([0.017, 0.571, 2.065])
IMG_SIZE = 512

DEPTH_WEIGHT = 1.0         # Metric depth is in meters — full weight is safe
SMOOTH_ITER = 3
MIN_SCALE = 0.10           # Only used as a sanity floor if regression is degenerate

def get_depth_map(image_path):
    print("[*] Generating METRIC depth map with Depth Anything V2 Metric Indoor Small...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        device="cpu"
    )
    img = Image.open(image_path).convert("RGB")
    depth = depth_pipe(img)["depth"]
    depth = np.array(depth, dtype=np.float32)
    depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))

    print("\n========== DEBUG: DEPTH MAP (METRIC) ==========")
    print(f"Shape: {depth.shape}")
    print(f"Depth min (m): {depth.min():.4f}")
    print(f"Depth max (m): {depth.max():.4f}")
    print(f"Depth mean (m): {depth.mean():.4f}")
    print(f"Depth std (m): {depth.std():.4f}")

    # Light smoothing — do NOT normalize, values are already in meters
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.05, sigmaSpace=10)

    # Save visualization (normalized only for viewing)
    depth_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_DEPTH_VIS, depth_vis)
    return depth

def refine_mesh(mesh_path, image_path):
    # 1. Load original MHR mesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    original_verts = mesh.vertices.copy()
    mesh.fix_normals()
    normals = mesh.vertex_normals
    depth_map = get_depth_map(image_path)

    # 2. Camera convention (confirmed by diagnostic)
    x_view = original_verts[:, 0] + CAM_T[0]
    y_view = original_verts[:, 1] + CAM_T[1]
    z_view = original_verts[:, 2] - CAM_T[2]

    distance = -z_view
    valid_mask = distance > 0.1

    u = (x_view / np.where(valid_mask, distance, 1.0)) * FOCAL_LENGTH + CX
    v = -(y_view / np.where(valid_mask, distance, 1.0)) * FOCAL_LENGTH + CY

    print("\n========== DEBUG: PROJECTION ==========")
    print(f"u range (valid): {u[valid_mask].min():.2f} to {u[valid_mask].max():.2f}")
    print(f"v range (valid): {v[valid_mask].min():.2f} to {v[valid_mask].max():.2f}")

    valid_mask &= (u >= 0) & (u < IMG_SIZE) & (v >= 0) & (v < IMG_SIZE)
    print(f"Valid vertices: {valid_mask.sum()} / {len(original_verts)}")

    if valid_mask.sum() < 100:
        print("[ERROR] Too few valid vertices, copying mesh unchanged.")
        mesh.export(OUTPUT_MESH)
        return

    u_idx = np.clip(u.astype(int), 0, IMG_SIZE - 1)
    v_idx = np.clip(v.astype(int), 0, IMG_SIZE - 1)
    sampled_depth = depth_map[v_idx, u_idx]

    print("\n========== DEBUG: SAMPLED DEPTH ==========")
    sd_valid = sampled_depth[valid_mask]
    print(f"Sampled (valid) min: {sd_valid.min():.4f}")
    print(f"Sampled (valid) max: {sd_valid.max():.4f}")
    print(f"Sampled (valid) mean: {sd_valid.mean():.4f}")

    # 3. Regression: metric depth (meters) -> mesh z_view (meters)
    # Both are linear, so a should be ~1.0
    mesh_z_valid = z_view[valid_mask]
    A = np.vstack([sampled_depth[valid_mask], np.ones(valid_mask.sum())]).T
    a, b = np.linalg.lstsq(A, mesh_z_valid, rcond=None)[0]

    print("\n========== DEBUG: REGRESSION ==========")
    print(f"Raw regression: a = {a:.6f}, b = {b:.6f}")

    # Safety: if regression collapses, use a fallback scale
    if abs(a) < MIN_SCALE:
        print(f"[WARN] Regression too small ({a:.6f}). Using fallback a = -1.0")
        a = -1.0  # camera looks down -Z, so larger depth (farther) = more negative z
        b = mesh_z_valid.mean() - a * sd_valid.mean()
        print(f"Fallback: a = {a:.6f}, b = {b:.6f}")

    target_z = a * sampled_depth + b
    target_z[~valid_mask] = z_view[~valid_mask]

    raw_z_disp = (target_z - z_view) * DEPTH_WEIGHT
    raw_z_disp = np.clip(raw_z_disp, -0.20, 0.20)  # clamp to 20 cm

    print("\n========== DEBUG: DISPLACEMENT ==========")
    print(f"min: {raw_z_disp.min():.6f}")
    print(f"max: {raw_z_disp.max():.6f}")
    print(f"mean: {raw_z_disp.mean():.6f}")
    print(f"std: {raw_z_disp.std():.6f}")

    # 4. Apply displacement in camera space
    refined_verts = original_verts.copy()
    refined_verts[:, 2] = z_view + raw_z_disp + CAM_T[2]

    # 5. Blend front/back based on facing direction
    facing_camera = np.clip(-normals[:, 2], 0, 1)
    facing_camera = facing_camera ** 1.5

    final_verts = (original_verts * (1 - facing_camera[:, None]) +
                   refined_verts * facing_camera[:, None])

    # 6. Back vertices stay as MHR template
    back_mask = facing_camera < 0.5
    print(f"\nKept {back_mask.sum()} back vertices at MHR template shape (no inflation)")

    # 7. Build and smooth
    refined_mesh = trimesh.Trimesh(vertices=final_verts, faces=mesh.faces)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.1, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"\n[OK] Exported refined mesh to {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
