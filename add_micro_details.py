import os
import glob
import numpy as np
import trimesh
import xatlas
import cv2

# --- CONFIGURATION ---
INPUT_MESH = "out_hmr2/hmr2_female_mesh_test-2.obj"
INPUT_DIR = "inputs2"
OUTPUT_MESH = "out_hmr2/hmr2_female_mesh_detailed.obj"
DISPLACEMENT_STRENGTH = 0.015  # 0.01 = subtle, 0.03 = strong

def find_input_image():
    exts = (".jpg", ".jpeg", ".jpge", ".png", ".bmp", ".webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(INPUT_DIR, "*" + ext)))
        files.extend(glob.glob(os.path.join(INPUT_DIR, "*" + ext.upper())))
    files = [f for f in files if not os.path.basename(f).startswith(".")]
    if not files:
        raise FileNotFoundError(f"No image found in {INPUT_DIR}/")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def generate_displacement_map(image_path, output_path):
    print(f"[*] Generating displacement map from {image_path}...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (1024, 1024))

    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x**2 + sobel_y**2)

    disp = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    disp = cv2.GaussianBlur(disp, (5, 5), 0)

    cv2.imwrite(output_path, (disp * 255).astype(np.uint8))
    print(f"[OK] Displacement map saved: {output_path}")
    return disp

def apply_details_to_mesh(mesh_path, disp_map, strength):
    print(f"[*] Loading mesh: {mesh_path}")
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    print(f"[*] Mesh loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    print("[*] UV unwrapping with xatlas...")
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    print(f"[OK] UV map created: {len(uvs)} UVs")

    h, w = disp_map.shape
    new_verts = mesh.vertices.copy()
    normals = mesh.vertex_normals

    print("[*] Applying displacement...")
    for i, uv in enumerate(uvs):
        u = int(np.clip(uv[0] * (w - 1), 0, w - 1))
        v = int(np.clip((1.0 - uv[1]) * (h - 1), 0, h - 1))
        d = (disp_map[v, u] - 0.5) * strength
        orig_idx = vmapping[i]
        new_verts[orig_idx] += normals[orig_idx] * d

    detailed = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)
    return detailed

def main():
    image_path = find_input_image()
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"[INFO] Input image: {image_path}")

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)

    disp_path = os.path.join(os.path.dirname(OUTPUT_MESH), f"{base_name}_displacement.png")
    disp_map = generate_displacement_map(image_path, disp_path)

    detailed_mesh = apply_details_to_mesh(INPUT_MESH, disp_map, DISPLACEMENT_STRENGTH)

    detailed_mesh.export(OUTPUT_MESH)
    size_kb = os.path.getsize(OUTPUT_MESH) / 1024
    print(f"[OK] Exported: {OUTPUT_MESH} ({size_kb:.1f} KB)")

    # Also export as GLB for easy viewing
    glb_path = OUTPUT_MESH.replace(".obj", ".glb")
    detailed_mesh.export(glb_path)
    print(f"[OK] Exported: {glb_path}")

if __name__ == "__main__":
    main()
