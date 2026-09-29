import os
import numpy as np
import cv2
import trimesh
from PIL import Image
from transformers import pipeline

INPUT_MESH = "out_sam/test-2_mesh.obj"
INPUT_IMAGE = "inputs/test-2.png"
OUTPUT_MESH = "out_sam/test-2_mesh_refined.obj"
OUTPUT_DEPTH_VIS = "out_sam/test-2_depth_vis.png"

FOCAL_LENGTH = 718.9
CX, CY = 256.0, 256.0
CAM_T = np.array([0.017, 0.571, 2.065])
IMG_SIZE = 512
DEPTH_FORCE = 0.10
SMOOTH_ITER = 3

def get_depth_map(image_path):
    print("[*] Generating depth map...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu"
    )
    img = Image.open(image_path).convert("RGB")
    depth = depth_pipe(img)["depth"]
    depth = np.array(depth, dtype=np.float32)

    print("\n========== DEBUG: RAW DEPTH ==========")
    print(f"Raw min: {depth.min():.4f}, max: {depth.max():.4f}, mean: {depth.mean():.4f}")

    depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.1, sigmaSpace=10)

    # Normalize to [0, 1] — pipeline may return uint8 or float; this handles both
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    print(f"Normalized min: {depth.min():.4f}, max: {depth.max():.4f}, mean: {depth.mean():.4f}")

    cv2.imwrite(OUTPUT_DEPTH_VIS, (depth * 255).astype(np.uint8))
    return depth

def refine_mesh(mesh_path, image_path):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    original_verts = mesh.vertices.copy()
    mesh.fix_normals()
    normals = mesh.vertex_normals
    depth_map = get_depth_map(image_path)

    x_view = original_verts[:, 0] + CAM_T[0]
    y_view = original_verts[:, 1] + CAM_T[1]
    z_view = original_verts[:, 2] - CAM_T[2]

    distance = -z_view
    valid_mask = distance > 0.1

    u = (x_view / np.where(valid_mask, distance, 1.0)) * FOCAL_LENGTH + CX
    v = -(y_view / np.where(valid_mask, distance, 1.0)) * FOCAL_LENGTH + CY
    valid_mask &= (u >= 0) & (u < IMG_SIZE) & (v >= 0) & (v < IMG_SIZE)

    u_idx = np.clip(u.astype(int), 0, IMG_SIZE - 1)
    v_idx = np.clip(v.astype(int), 0, IMG_SIZE - 1)
    sampled_depth = depth_map[v_idx, u_idx]

    # --- AUTO-DETECT SIGN VIA CORRELATION ---
    corr = np.corrcoef(sampled_depth[valid_mask], z_view[valid_mask])[0, 1]
    print(f"\n========== DEBUG: CORRELATION ==========")
    print(f"corr(depth, z_view) = {corr:.4f}")

    # We want: high depth → more negative z_view (closer). That means corr < 0.
    # If corr > 0, the depth map is inverted. Flip it.
    if corr > 0:
        print("[FIX] Depth map inverted. Flipping.")
        depth_map = 1.0 - depth_map
        sampled_depth = depth_map[v_idx, u_idx]

    # --- DIRECT SIGNED DISPLACEMENT (no regression) ---
    depth_centered = sampled_depth - 0.5
    # Higher depth (closer) → pull forward (more negative z_view)
    displacement = -depth_centered * 2.0 * DEPTH_FORCE

    print(f"\n========== DEBUG: DISPLACEMENT ==========")
    print(f"min: {displacement.min():.5f}, max: {displacement.max():.5f}")
    print(f"mean: {displacement.mean():.5f}, std: {displacement.std():.5f}")

    new_z_view = z_view + displacement
    refined_verts = original_verts.copy()
    refined_verts[:, 2] = new_z_view + CAM_T[2]

    facing_camera = np.clip(-normals[:, 2], 0, 1) ** 1.5
    final_verts = (original_verts * (1 - facing_camera[:, None]) +
                   refined_verts * facing_camera[:, None])

    refined_mesh = trimesh.Trimesh(vertices=final_verts, faces=mesh.faces)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.15, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"\n[OK] Exported refined mesh to {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
