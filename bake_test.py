import os
import glob
import numpy as np
import trimesh
import xatlas
from PIL import Image, ImageDraw

def find_files(input_dir):
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    mesh_exts = ['*.glb', '*.obj', '*.ply', '*.stl']
    img_files = [f for ext in img_exts for f in glob.glob(os.path.join(input_dir, ext))]
    mesh_files = [f for ext in mesh_exts for f in glob.glob(os.path.join(input_dir, ext))]
    if not img_files or not mesh_files:
        raise FileNotFoundError(f"Missing image or mesh in {input_dir}")
    return sorted(img_files)[0], sorted(mesh_files)[0]

def barycentric_interpolate(p0, p1, p2, x, y):
    """Calculates barycentric weights for a point (x, y) inside a triangle."""
    det = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if det == 0:
        return 0, 0, 0
    l1 = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / det
    l2 = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / det
    l3 = 1 - l1 - l2
    return l1, l2, l3

def bake_texture(image_path, mesh_path, output_dir, texture_size=2048):
    print(f"[*] Loading Image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_pixels = np.array(img)
    
    print(f"[*] Loading Mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')
    
    print("[*] Generating UV Maps using xatlas (CPU)...")
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    
    print("[*] Projecting 3D vertices to 2D Image space...")
    bounds = mesh.bounds
    min_v, max_v = bounds[0, :2], bounds[1, :2]
    scale = np.array([W, H]) / (max_v - min_v + 1e-6)
    
    # Project 3D vertices to 2D image coordinates
    proj_v = (mesh.vertices[:, :2] - min_v) * scale
    proj_v[:, 1] = H - proj_v[:, 1] # Flip Y
    proj_v = np.clip(proj_v, [0, 0], [W-1, H-1]).astype(int)
    
    # Sample colors from the image for each vertex
    vertex_colors = img_pixels[proj_v[:, 1], proj_v[:, 0]]
    
    print(f"[*] Baking Texture Atlas ({texture_size}x{texture_size}) with Barycentric Interpolation...")
    texture = Image.new("RGB", (texture_size, texture_size), (255, 255, 255))
    draw = ImageDraw.Draw(texture)
    uv_pixels = (uvs * (texture_size - 1)).astype(int)
    
    # Convert to PIL coordinates for faster drawing
    for face_idx in range(len(mesh.faces)):
        v_idx = mesh.faces[face_idx]
        uv_tri = uv_pixels[v_idx]
        colors = vertex_colors[v_idx]
        
        # Get bounding box of the triangle in UV space to limit iteration
        min_x, max_x = min(uv_tri[:, 0]), max(uv_tri[:, 0])
        min_y, max_y = min(uv_tri[:, 1]), max(uv_tri[:, 1])
        
        # Skip tiny triangles
        if max_x - min_x < 1 or max_y - min_y < 1:
            continue
            
        # Rasterize the triangle with interpolation
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                l1, l2, l3 = barycentric_interpolate(uv_tri[0], uv_tri[1], uv_tri[2], x, y)
                if l1 >= 0 and l2 >= 0 and l3 >= 0:
                    # Interpolate color
                    color = l1 * colors[0] + l2 * colors[1] + l3 * colors[2]
                    draw.point((x, y), fill=tuple(color.astype(int)))
    
    print("[*] Applying texture to mesh...")
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    mesh.visual.material.image = texture
    
    os.makedirs(output_dir, exist_ok=True)
    out_glb_path = os.path.join(output_dir, "processed_model.glb")
    mesh.export(out_glb_path)
    print(f"[+] Success! Exported GLB to: {out_glb_path}")

if __name__ == "__main__":
    INPUT_DIR = "inputs"
    OUTPUT_DIR = "out_fusion"
    os.makedirs(INPUT_DIR, exist_ok=True)
    try:
        img_path, mesh_path = find_files(INPUT_DIR)
        bake_texture(img_path, mesh_path, OUTPUT_DIR)
    except Exception as e:
        print(f"[-] Error: {e}")
