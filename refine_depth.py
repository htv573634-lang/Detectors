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

# Camera parameters from your SAM3DBody log
FOCAL_LENGTH = 718.9
CX, CY = 256.0, 256.0   # Image center (512x512)
CAM_T = np.array([0.017, 0.571, 2.065]) 
IMG_SIZE = 512

# Displacement tuning
DEPTH_WEIGHT = 0.4       # Higher weight since we are now masking properly
SMOOTH_ITER = 3

def get_depth_map(image_path):
    print("[*] Generating depth map with Depth Anything V2...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu"
    )
    img = Image.open(image_path).convert("RGB")
    depth = depth_pipe(img)["depth"]
    depth = np.array(depth, dtype=np.float32)
    depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
    
    # Normalize 0-1
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Smooth the depth map to remove concentric rings
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.1, sigmaSpace=10)
    
    depth_vis = (depth * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_DEPTH_VIS, depth_vis)
    return depth

def refine_mesh(mesh_path, image_path):
    # 1. Load mesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    print(f"[*] Loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # 2. Get smoothed depth map
    depth_map = get_depth_map(image_path)

    # 3. Project mesh vertices to 2D (Perspective projection)
    verts = mesh.vertices.copy()
    verts_cam = verts - CAM_T
    
    z_safe = np.clip(verts_cam[:, 2], 0.01, None)
    u = (verts_cam[:, 0] / z_safe) * FOCAL_LENGTH + CX
    v = -(verts_cam[:, 1] / z_safe) * FOCAL_LENGTH + CY
    
    u_idx = np.clip(u.astype(int), 0, IMG_SIZE - 1)
    v_idx = np.clip(v.astype(int), 0, IMG_SIZE - 1)
    
    sampled_depth = depth_map[v_idx, u_idx]
    
    # 4. Align Depth to Mesh Z
    mesh_z = verts_cam[:, 2]
    A = np.vstack([sampled_depth, np.ones(len(sampled_depth))]).T
    a, b = np.linalg.lstsq(A, mesh_z, rcond=None)[0]
    target_z = a * sampled_depth + b
    
    # 5. Compute raw displacement
    raw_z_disp = (target_z - mesh_z) * DEPTH_WEIGHT
    
    # 6. CRITICAL FIX: Mask displacement to front-facing vertices only
    # Get vertex normals in camera space (normals don't need translation, only rotation)
    # The MHR model provides normals, but we can calculate them if missing.
    mesh.fix_normals()
    normals = mesh.vertex_normals
    
    # In camera space, the camera looks down the -Z axis. 
    # Front-facing vertices have normals pointing towards the camera (negative Z component).
    # We use the dot product of the normal with the camera direction [0,0,-1]
    # to get a weight between 0 and 1.
    facing_camera = np.clip(-normals[:, 2], 0, 1)
    
    # Apply the mask: Only front-facing vertices get displaced.
    masked_z_disp = raw_z_disp * facing_camera
    
    # 7. Apply displacement
    refined_verts = verts.copy()
    refined_verts[:, 2] += masked_z_disp
    
    # 8. Build new mesh and smooth lightly
    refined_mesh = trimesh.Trimesh(vertices=refined_verts, faces=mesh.faces)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.1, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"[OK] Exported refined mesh to {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
