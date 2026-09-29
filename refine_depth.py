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

DEPTH_WEIGHT = 0.5
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
    depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.1, sigmaSpace=10)
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

    # Project to 2D
    verts_cam = original_verts - CAM_T
    z_safe = np.clip(verts_cam[:, 2], 0.01, None)
    u = (verts_cam[:, 0] / z_safe) * FOCAL_LENGTH + CX
    v = -(verts_cam[:, 1] / z_safe) * FOCAL_LENGTH + CY
    
    # --- DEBUG 1: Projection Stats ---
    print("\n========== DEBUG: PROJECTION ==========")
    print(f"Projected U range: {u.min():.2f} to {u.max():.2f}")
    print(f"Projected V range: {v.min():.2f} to {v.max():.2f}")
    
    # FIX 1: Create a valid mask to exclude bad projections
    valid_mask = (u >= 0) & (u < IMG_SIZE) & (v >= 0) & (v < IMG_SIZE) & (z_safe > 0.5)
    print(f"Valid vertices for depth sampling: {valid_mask.sum()} / {len(verts_cam)}")
    
    u_idx = np.clip(u.astype(int), 0, IMG_SIZE - 1)
    v_idx = np.clip(v.astype(int), 0, IMG_SIZE - 1)
    
    sampled_depth = depth_map[v_idx, u_idx]
    
    # FIX 2: Normalize sampled depth to 0-1 range
    sd_valid = sampled_depth[valid_mask]
    sd_min, sd_max = sd_valid.min(), sd_valid.max()
    sampled_depth_norm = (sampled_depth - sd_min) / (sd_max - sd_min + 1e-8)
    
    print(f"Sampled depth range (valid): {sd_min:.4f} to {sd_max:.4f}")

    # --- DEBUG 2: Linear Regression on valid vertices only ---
    print("\n========== DEBUG: LINEAR REGRESSION ==========")
    mesh_z_valid = verts_cam[valid_mask, 2]
    
    A = np.vstack([sampled_depth_norm[valid_mask], np.ones(valid_mask.sum())]).T
    a, b = np.linalg.lstsq(A, mesh_z_valid, rcond=None)[0]
    
    print(f"Coefficient a (scale): {a:.6f}")
    print(f"Coefficient b (shift): {b:.6f}")
    print(f"Interpretation: target_z = {a:.4f} * depth_norm + {b:.4f}")
    
    target_z = a * sampled_depth_norm + b
    
    # Keep original Z for invalid vertices
    target_z[~valid_mask] = verts_cam[~valid_mask, 2]
    
    raw_z_disp = (target_z - verts_cam[:, 2]) * DEPTH_WEIGHT
    raw_z_disp = np.clip(raw_z_disp, -0.05, 0.05)
    
    # --- DEBUG 3: Displacement Stats ---
    print("\n========== DEBUG: DISPLACEMENT ==========")
    print(f"Raw Z displacement min: {raw_z_disp.min():.6f}")
    print(f"Raw Z displacement max: {raw_z_disp.max():.6f}")
    print(f"Raw Z displacement mean: {raw_z_disp.mean():.6f}")
    print(f"Raw Z displacement std: {raw_z_disp.std():.6f}")

    depth_refined_verts = original_verts.copy()
    depth_refined_verts[:, 2] += raw_z_disp

    # FIX 3: Masking normals to preserve back volume
    facing_camera = np.clip(-normals[:, 2], 0, 1)
    facing_camera = facing_camera ** 2.0 
    
    # Ensure invalid vertices keep original Z
    facing_camera[~valid_mask] = 0.0
    
    final_verts = (original_verts * (1 - facing_camera[:, None]) + 
                   depth_refined_verts * facing_camera[:, None])
    
    refined_mesh = trimesh.Trimesh(vertices=final_verts, faces=mesh.faces)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.1, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"\n[OK] Exported refined mesh to {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
