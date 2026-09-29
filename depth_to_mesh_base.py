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
MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
IMG_SIZE = 512
FOCAL_LENGTH = IMG_SIZE * 0.8
CX, CY = IMG_SIZE / 2.0, IMG_SIZE / 2.0

def generate_mesh():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Loading Depth Anything V2 Base model ({MODEL_ID})...")

    depth_pipe = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device="cpu"
    )

    print(f"[*] Processing image: {INPUT_IMAGE}")
    img = Image.open(INPUT_IMAGE).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    depth_result = depth_pipe(img_resized)
    depth_map = np.array(depth_result["depth"], dtype=np.float32)

    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

    print(f"[*] Depth map generated. Range: {depth_min:.4f} to {depth_max:.4f}")

    u, v = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))

    z_scale = 2.0
    z = depth_norm * z_scale + 0.5

    x = (u - CX) * z / FOCAL_LENGTH
    y = (v - CY) * z / FOCAL_LENGTH

    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

    colors = np.array(img_resized).reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if len(pcd.points) > 100000:
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

    print(f"[*] Point cloud created with {len(pcd.points)} points.")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    print("[*] Running Poisson reconstruction...")
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
    poisson_mesh.compute_vertex_normals()

    output_path = os.path.join(OUTPUT_DIR, "depth_mesh.obj")
    o3d.io.write_triangle_mesh(output_path, poisson_mesh)
    print(f"[*] Mesh exported to {output_path}")

    pcd.export(os.path.join(OUTPUT_DIR, "point_cloud.ply"))

if __name__ == "__main__":
    generate_mesh()
