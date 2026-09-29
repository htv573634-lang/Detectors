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

DEPTH_WEIGHT = 0.6
SMOOTH_ITER = 3

def get_depth_map(image_path):
    print("[*] Generating depth map with Depth Anything V2 Small (relative)...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu"
    )
    img = Image.open(image_path).convert("RGB")
    depth = depth_pipe(img)["depth"]
    depth = np.array(depth, dtype=np.float32)
    depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.1, sigmaSpace=10)

    print("\n========== DEBUG: DEPTH MAP ==========")
    print(f"Depth min: {depth.min():.4f}")
    print(f"Depth max: {depth.max():.4f}")
    print(f"Depth mean: {depth.mean():.4f}")

    depth_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_DEPTH_VIS, depth_vis)
    return depth

def refine_mesh(mesh_path, image_path):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    original_verts = mesh.vertices.copy()
    mesh.fix_normals()
    normals = mesh.vertex_normals
    depth_map = get_depth_map(image_path)

    # Camera projection (confirmed by diagnostic)
    x_view = original_verts[:, 0] + CAM_T[0]
    y_view = original_verts[:, 1] + CAM_T[1]
    z_view = original_verts[:, 2] - CAM_T[2]

    distance = -z_view
    valid_mask = distance > 0.1

    u = (x_view / np.where(valid_mask, distance, 1.0)) * FOCAL_LENGTH + CX
    v = -(y_view / np.where(valid_mask, distance, 1.0)) * FOCAL_LENGTH + CY

    valid_mask &= (u >= 0) & (u < IMG_SIZE) & (v >= 0) & (v < IMG_SIZE)
    print(f"\nValid vertices: {valid_mask.sum()} / {len(original_verts)}")

    if valid_mask.sum() < 100:
        print("[ERROR] Too few valid vertices.")
        mesh.export(OUTPUT_MESH)
        return

    u_idx = np.clip(u.astype(int), 0, IMG_SIZE - 1)
    v_idx = np.clip(v.astype(int), 0, IMG_SIZE - 1)
    sampled_depth = depth_map[v_idx, u_idx]

    # --- NORMALIZED REGRESSION (the key fix) ---
    # Step 1: normalize sampled depth to [0, 1] over valid vertices only
    sd_valid = sampled_depth[valid_mask]
    sd_min, sd_max = sd_valid.min(), sd_valid.max()
    sd_norm = (sampled_depth - sd_min) / (sd_max - sd_min + 1e-8)

    # Step 2: normalize mesh z_view to [0, 1] over valid vertices only
    mv_valid = z_view[valid_mask]
    mv_min, mv_max = mv_valid.min(), mv_valid.max()
    mv_norm = (z_view - mv_min) / (mv_max - mv_min + 1e-8)

    # Step 3: regression in normalized space (both in [0,1])
    A = np.vstack([sd_norm[valid_mask], np.ones(valid_mask.sum())]).T
    a_norm, b_norm = np.linalg.lstsq(A, mv_norm[valid_mask], rcond=None)[0]

    print("\n========== DEBUG: REGRESSION (NORMALIZED) ==========")
    print(f"a_norm = {a_norm:.4f}  (target ±1.0)")
    print(f"b_norm = {b_norm:.4f}  (target ≈ 0)")
    print(f"Mesh z_view range: {mv_min:.4f} to {mv_max:.4f} ({(mv_max-mv_min)*100:.2f} cm)")
    print(f"Depth range: {sd_min:.4f} to {sd_max:.4f}")

    # Step 4: map predicted normalized z back to real mesh z scale
    target_z_norm = a_norm * sd_norm + b_norm
    target_z = target_z_norm * (mv_max - mv_min) + mv_min
    target_z[~valid_mask] = z_view[~valid_mask]

    # Step 5: displacement (in meters, already scaled)
    raw_z_disp = (target_z - z_view) * DEPTH_WEIGHT
    raw_z_disp = np.clip(raw_z_disp, -0.08, 0.08)  # clamp to 8 cm

    print("\n========== DEBUG: DISPLACEMENT ==========")
    print(f"min: {raw_z_disp.min():.5f}  max: {raw_z_disp.max():.5f}")
    print(f"mean: {raw_z_disp.mean():.5f}  std: {raw_z_disp.std():.5f}")

    # Step 6: apply displacement
    refined_verts = original_verts.copy()
    refined_verts[:, 2] = z_view + raw_z_disp + CAM_T[2]

    # Step 7: blend based on facing camera
    facing_camera = np.clip(-normals[:, 2], 0, 1) ** 1.5
    final_verts = (original_verts * (1 - facing_camera[:, None]) +
                   refined_verts * facing_camera[:, None])

    back_mask = facing_camera < 0.5
    print(f"Kept {back_mask.sum()} back vertices at MHR template shape")

    # Step 8: build and smooth
    refined_mesh = trimesh.Trimesh(vertices=final_verts, faces=mesh.faces)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.15, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"\n[OK] Exported refined mesh to {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
