import os
import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import pipeline

# --- CONFIGURATION ---
INPUT_IMAGE = "inputs/test-2.png"
OUTPUT_DIR = "out_depth_mesh"
# Use the Base model for better accuracy (can also use 'large' if CPU memory allows)
MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
# Approximate focal length (adjust based on your image's field of view)
# A value of ~0.8 * image_width is a reasonable starting point for a 60° FOV
IMG_SIZE = 512
FOCAL_LENGTH = IMG_SIZE * 0.8 
CX, CY = IMG_SIZE / 2.0, IMG_SIZE / 2.0

def generate_mesh():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Loading Depth Anything V2 Base model ({MODEL_ID})...")
    
    # 1. Load the pipeline
    depth_pipe = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device="cpu"
    )
    
    # 2. Load and preprocess image
    print(f"[*] Processing image: {INPUT_IMAGE}")
    img = Image.open(INPUT_IMAGE).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    
    # 3. Get depth map
    depth_result = depth_pipe(img_resized)
    depth_map = np.array(depth_result["depth"], dtype=np.float32)
    
    # Normalize depth to 0-1 range for stable back-projection
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    
    print(f"[*] Depth map generated. Range: {depth_min:.4f} to {depth_max:.4f}")
    
    # 4. Back-project to 3D Point Cloud
    # Create pixel grid
    u, v = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
    
    # Convert depth (0-1 relative) to a plausible Z-range (e.g., 0.5 to 2.5 meters)
    # This is a guess since Depth Anything outputs relative depth
    z_scale = 2.0  # Max depth in meters
    z = depth_norm * z_scale + 0.5
    
    # Unproject using pinhole camera model
    x = (u - CX) * z / FOCAL_LENGTH
    y = (v - CY) * z / FOCAL_LENGTH
    
    # Stack into Nx3 point cloud
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    
    # Get colors from original image for visualization
    colors = np.array(img_resized).reshape(-1, 3) / 255.0
    
    # 5. Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Downsample if too large (Poisson is heavy)
    if len(pcd.points) > 100000:
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
    
    print(f"[*] Point cloud created with {len(pcd.points)} points.")
    
    # 6. Estimate Normals (required for Poisson)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    # Orient normals towards the camera (viewpoint)
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    # 7. Poisson Surface Reconstruction
    print("[*] Running Poisson reconstruction...")
    # Depth 8-9 is typical. Lower = smoother, Higher = more detail (and memory)
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    
    # 8. Remove low-density artifacts (flying vertices)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
    poisson_mesh.compute_vertex_normals()
    
    # 9. Export
    output_path = os.path.join(OUTPUT_DIR, "depth_mesh.obj")
    o3d.io.write_triangle_mesh(output_path, poisson_mesh)
    print(f"[*] Mesh exported to {output_path}")
    
    # Also export as PLY with colors for easier viewing
    pcd.export(os.path.join(OUTPUT_DIR, "point_cloud.ply"))

if __name__ == "__main__":
    generate_mesh()a
