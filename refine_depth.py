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

DEPTH_WEIGHT = 0.8         # Amplified since we're forcing min scale
SMOOTH_ITER = 3
MIN_SCALE = 0.15           # Force regression scale to be at least this
BACK_INFLATE = 0.03        # Push back surface out by 3cm for volume

def get_depth_map(image_path):
    print("[*] Generating depth map with Depth Anything V2 Small...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu"
    )
    img = Image.open(image_path).convert("RGB")
    depth = depth_pipe(img)["depth"]
    depth = np.array(depth, dtype=np.float32)
    depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.1, sigmaSpace=10)
    cv2.imwrite(OUTPUT_DEPTH_VIS, (depth * 255).astype(np.uint8))
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

    # 2. Correct camera convention (from diagnostic)
    # x_view = x + cam_t[0]
    # y_view = y + cam_t[1]
    # z_view = z - cam_t[2]  (negative = in front)
    # distance = -z_view
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

    # 3. Normalize sampled depth using valid vertices
    sd_valid = sampled_depth[valid_mask]
    sd_min, sd_max = sd_valid.min(), sd_valid.max()
    sampled_depth_norm = (sampled_depth - sd_min) / (sd_max - sd_min + 1e-8)

    # 4. Regression with FORCED minimum scale
    mesh_z_valid = z_view[valid_mask]
    A = np.vstack([sampled_depth_norm[valid_mask], np.ones(valid_mask.sum())]).T
    a, b = np.linalg.lstsq(A, mesh_z_valid, rcond=None)[0]

    print("\n========== DEBUG: REGRESSION ==========")
    print(f"Raw regression: a = {a:.6f}, b = {b:.6f}")

    # Force minimum scale so depth map actually affects the mesh
    sign_a = -1.0 if a < 0 else 1.0
    if abs(a) < MIN_SCALE:
        a = sign_a * MIN_SCALE
        print(f"Forced minimum: a = {a:.6f}")
    else:
        print(f"Kept original: a = {a:.6f}")

    target_z = a * sampled_depth_norm + b
    target_z[~valid_mask] = z_view[~valid_mask]

    raw_z_disp = (target_z - z_view) * DEPTH_WEIGHT
    raw_z_disp = np.clip(raw_z_disp, -0.15, 0.15)

    print("\n========== DEBUG: DISPLACEMENT ==========")
    print(f"min: {raw_z_disp.min():.6f}, max: {raw_z_disp.max():.6f}")
    print(f"mean: {raw_z_disp.mean():.6f}, std: {raw_z_disp.std():.6f}")

    # 5. Apply front displacement
    refined_verts = original_verts.copy()
    refined_verts[:, 2] = z_view + raw_z_disp + CAM_T[2]

    # 6. Blend front/back with back volume inflation
    facing_camera = np.clip(-normals[:, 2], 0, 1)  # 1.0=front, 0.0=back
    facing_camera = facing_camera ** 1.5  # Smoother transition

    final_verts = (original_verts * (1 - facing_camera[:, None]) +
                   refined_verts * facing_camera[:, None])

    # 7. Back volume inflation (push back of body outward)
    back_mask = facing_camera < 0.5
    final_verts[back_mask, 2] += BACK_INFLATE * (1.0 - facing_camera[back_mask])
    print(f"Inflated {back_mask.sum()} back vertices by up to {BACK_INFLATE*100:.1f}cm")

    # 8. Build and smooth
    refined_mesh = trimesh.Trimesh(vertices=final_verts, faces=mesh.faces)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.1, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"\n[OK] Exported refined mesh to {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
