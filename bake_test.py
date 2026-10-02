import os
import glob
import numpy as np
import trimesh
import xatlas
from PIL import Image, ImageDraw

def find_files(input_dir):
    """Finds the first valid image and mesh file in the input directory."""
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    mesh_exts = ['*.glb', '*.obj', '*.ply', '*.stl']
    
    img_files = []
    mesh_files = []
    
    for ext in img_exts:
        img_files.extend(glob.glob(os.path.join(input_dir, ext)))
    for ext in mesh_exts:
        mesh_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not img_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    if not mesh_files:
        raise FileNotFoundError(f"No mesh files found in {input_dir}")
        
    # Sort to ensure deterministic behavior
    return sorted(img_files)[0], sorted(mesh_files)[0]

def bake_texture(image_path, mesh_path, output_dir, texture_size=2048):
    print(f"[*] Loading Image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_pixels = np.array(img)
    
    print(f"[*] Loading Mesh: {mesh_path}")
    # Force load as a single mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    print("[*] Generating UV Maps using xatlas (CPU)...")
    # xatlas.parametrize returns: vmapping, indices, uvs
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    
    # Apply the new UV vertices and faces back to the mesh
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    
    print("[*] Projecting 3D vertices to 2D Image space...")
    # Get bounding box of the mesh
    bounds = mesh.bounds
    min_v = bounds[0, :2]
    max_v = bounds[1, :2]
    
    # Calculate scale to fit the image bounding box
    scale = np.array([W, H]) / (max_v - min_v + 1e-6)
    
    # Project 3D vertices (X, Y) to 2D image coordinates
    proj_v = (mesh.vertices[:, :2] - min_v) * scale
    proj_v[:, 1] = H - proj_v[:, 1] # Flip Y axis (Image Y goes down, 3D Y goes up)
    
    # Clip to image boundaries and convert to integers
    proj_v = np.clip(proj_v, [0, 0], [W-1, H-1]).astype(int)
    
    # Sample colors from the original image for each vertex
    vertex_colors = img_pixels[proj_v[:, 1], proj_v[:, 0]]
    
    print(f"[*] Baking Texture Atlas ({texture_size}x{texture_size})...")
    # Create a blank white texture atlas
    texture = Image.new("RGB", (texture_size, texture_size), (255, 255, 255))
    draw = ImageDraw.Draw(texture)
    
    # Scale UVs to texture size
    uv_pixels = (uvs * (texture_size - 1)).astype(int)
    
    # Rasterize each triangle onto the texture atlas
    # Note: This uses flat shading (average color) for speed. 
    # For photorealistic results, barycentric interpolation is needed.
    for face_idx in range(len(mesh.faces)):
        v_idx = mesh.faces[face_idx]
        
        # Get the 3 UV coordinates for this face
        uv_tri = uv_pixels[v_idx]
        
        # Get the average color of the 3 vertices for this face
        avg_color = tuple(np.mean(vertex_colors[v_idx], axis=0).astype(int))
        
        # Draw the triangle onto the texture
        draw.polygon([tuple(pt) for pt in uv_tri], fill=avg_color)
    
    print("[*] Applying texture to mesh...")
    # Assign the baked texture and UVs to the mesh
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    mesh.visual.material.image = texture
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export as GLB (Best format for textures)
    out_glb_path = os.path.join(output_dir, "processed_model.glb")
    mesh.export(out_glb_path)
    print(f"[+] Success! Exported GLB to: {out_glb_path}")
    
    # Optional: Export as OBJ with a separate MTL and Texture file
    out_obj_path = os.path.join(output_dir, "processed_model.obj")
    mesh.export(out_obj_path)
    print(f"[+] Success! Exported OBJ to: {out_obj_path}")

if __name__ == "__main__":
    INPUT_DIR = "inputs"
    OUTPUT_DIR = "out_fusion"
    
    # Ensure inputs directory exists
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"[!] Created '{INPUT_DIR}' directory. Please place your image and mesh files there and run again.")
    else:
        try:
            img_path, mesh_path = find_files(INPUT_DIR)
            bake_texture(img_path, mesh_path, OUTPUT_DIR)
        except Exception as e:
            print(f"[-] Error: {e}")
