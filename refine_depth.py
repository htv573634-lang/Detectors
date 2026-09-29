import os
import numpy as np
import cv2
import trimesh
from PIL import Image
from transformers import pipeline

INPUT_MESH = "out_sam/test-2_mesh.obj"
INPUT_IMAGE = "inputs/test-2.png"
OUTPUT_MESH = "out_sam/test-2_mesh_refined.obj"
OUTPUT_NORMAL_MAP = "out_sam/test-2_normal_map.png"

# Camera parameters from SAM3DBody log
FOCAL_LENGTH = 718.9
IMG_SIZE = 512
DEPTH_WEIGHT = 0.85       # trust the depth map more
NORMAL_WEIGHT = 0.5       # how much to tilt normals
SMOOTH_ITER = 5

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
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth_norm

def depth_to_normal_map(depth):
    """Convert depth map to a normal map (curvature info)."""
    print("[*] Computing normal map from depth gradient...")
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)

    # Normalize gradient strength
    strength = 5.0
    nx = -grad_x * strength
    ny = -grad_y * strength
    nz = np.ones_like(depth)

    length = np.sqrt(nx*nx + ny*ny + nz*nz)
    nx /= length
    ny /= length
    nz /= length

    # Encode to RGB (0-255)
    rgb = np.stack([(nx + 1) * 0.5, (ny + 1) * 0.5, (nz + 1) * 0.5], axis=-1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # Save visualization
    cv2.imwrite(OUTPUT_NORMAL_MAP, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    print(f"[OK] Normal map saved: {OUTPUT_NORMAL_MAP}")

    return nx, ny, nz

def refine_mesh(mesh_path, image_path):
    # 1. Load mesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    print(f"[*] Loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # 2. Get depth + normal map
    depth_map = get_depth_map(image_path)
    nx, ny, nz = depth_to_normal_map(depth_map)

    # 3. Project mesh vertices to 2D (orthographic, front view)
    verts = mesh.vertices.copy()

    # Center and scale to image space
    center = verts.mean(axis=0)
    verts_c = verts - center
    extent = np.abs(verts_c).max()
    scale = (IMG_SIZE * 0.4) / extent
    verts_s = verts_c * scale

    u = np.clip((verts_s[:, 0] + IMG_SIZE/2).astype(int), 0, IMG_SIZE-1)
    v = np.clip((verts_s[:, 1] + IMG_SIZE/2).astype(int), 0, IMG_SIZE-1)

    # 4. Sample depth and normal at each vertex
    sampled_depth = depth_map[v, u]
    sampled_nx = nx[v, u]
    sampled_ny = ny[v, u]
    sampled_nz = nz[v, u]

    # 5. Compute per-vertex normals of the mesh (for displacement direction)
    mesh.fix_normals()
    vertex_normals = mesh.vertex_normals

    # 6. Compute displacement along the vertex normal, weighted by depth match
    mesh_z_min, mesh_z_max = verts_s[:, 2].min(), verts_s[:, 2].max()
    target_z = mesh_z_max - sampled_depth * (mesh_z_max - mesh_z_min)

    # Displacement magnitude (blend Z-match with normal direction)
    z_displacement = (target_z - verts_s[:, 2]) * DEPTH_WEIGHT

    # Apply displacement along vertex normal, scaled by the z-displacement
    # This gives curvature rather than flat pushing
    displaced = verts_s + vertex_normals * z_displacement[:, None] * 0.5
    # Also apply the pure Z push
    displaced[:, 2] = verts_s[:, 2] + z_displacement

    # 7. Rescale back to original coordinate frame
    refined_verts = (displaced / scale) + center

    # 8. Build new mesh
    refined_mesh = trimesh.Trimesh(vertices=refined_verts, faces=mesh.faces)

    # 9. Gentle smoothing (much less than before)
    trimesh.smoothing.filter_taubin(refined_mesh, lamb=0.2, nu=0.5, iterations=SMOOTH_ITER)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    refined_mesh.export(OUTPUT_MESH)
    print(f"[OK] Exported refined mesh: {OUTPUT_MESH}")

if __name__ == "__main__":
    refine_mesh(INPUT_MESH, INPUT_IMAGE)
