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

# LOWERED WEIGHT to prevent slicing the back
DEPTH_WEIGHT = 0.3      
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
    
    u_idx = np.clip(u.astype(int), 0, IMG_SIZE - 1)
    v_idx = np.clip(v.astype(int), 0, IMG_SIZE - 1)
    sampled_depth = depth_map[v_idx, u_idx]
    
    # Align Depth to Mesh Z
    mesh_z = verts_cam[:, 2]
    A = np.vstack([sampled_depth, np.ones(len(sampled_depth))]).T
    a, b = np.linalg.lstsq(A, mesh_z, rcond=None)[0]
    target_z = a * sampled_depth + b
    
    # Displacement with clamping
    raw_z_disp = (target_z - mesh_z) * DEPTH_WEIGHT
    raw_z_disp = np.clip(raw_z_disp, -0.03, 0.03)
    
    depth_refined_verts = original_verts.copy()
    depth_refined_verts[:, 2] += raw_z_disp

    # Volumetric Blending (smooth transition from front to back)
    facing_camera = np.clip(-normals[:, 2], 0, 1)
    # Apply a power curve to make the transition smoother, preserving volume on the sides
    facing_camera = facing_camera ** 2.0 
    
    final_verts = (original_verts * (1 - facing_camera[:, None]) + 
                   depth_refined_verts * facing_camera[:, None])
    
    refined_mesh = trimesh.Trimesh(vertices=final_verts, faces=mesh.faces)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.1, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"[OK] Exported refined mesh to {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
