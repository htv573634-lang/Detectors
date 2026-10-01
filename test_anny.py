import os
import glob
import sys
import numpy as np
import torch
import trimesh

# ── CONFIG ──
INPUT_DIR = "inputs"
OUTPUT_DIR = "out_triposr"
IMAGE_EXTS = (
    "jpg", "jpeg", "jpge", "jpe", "jfif",
    "png", "bmp", "webp", "tif", "tiff", "gif", "ppm"
)

def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
    return files

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = find_images()
    if not images:
        print(f"[ERROR] No images found in {INPUT_DIR}/")
        sys.exit(1)

    print(f"[*] Found {len(images)} image(s)")

    # ── Load Anny model (CPU, float32) ──
    print("[*] Loading Anny model on CPU...")
    import anny
    model = anny.Anny(
        local_changes="default",
        facial_actions="all",
    ).to(dtype=torch.float32, device="cpu")
    model.eval()
    print("[OK] Anny loaded")

    # ── Rest pose (identity transformation for every bone) ──
    pose_parameters = torch.eye(4, dtype=torch.float32)[None, None].repeat(
        1, model.bone_count, 1, 1
    )

    # ── Default phenotype (neutral adult) ──
    phenotype_kwargs = dict(
        gender=0.5,
        age=0.5,
        muscle=0.5,
        weight=0.5,
        height=0.5,
        proportions=0.5,
    )

    # ── Process each image ──
    for i, img_path in enumerate(images, 1):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{i}/{len(images)}] {img_path}")

        # Optional: read image just to confirm it exists and get dimensions
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                W, H = im.size
            print(f"    Image: {W}x{H}")
        except Exception as e:
            print(f"    [WARN] Cannot read image: {e}")

        # ── Generate body mesh ──
        with torch.no_grad():
            out = model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs,
                local_changes_kwargs={},
                pose_parameterization=None,
                return_bone_ends=False,
            )

        vertices = out["vertices"].squeeze(0).cpu().numpy()
        faces = model.faces.cpu().numpy()

        # ── Build trimesh ──
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        # Optional: apply Y-up rotation for standard 3D viewers
        try:
            import roma
            transform = (
                roma.Rigid(
                    linear=roma.euler_to_rotmat("x", [-90.0], degrees=True),
                    translation=None,
                )
                .to_homogeneous()
                .cpu()
                .numpy()
            )
            mesh.apply_transform(transform)
        except ImportError:
            pass  # roma is optional

        # ── Export ──
        obj_path = os.path.join(OUTPUT_DIR, f"{name}_anny.obj")
        glb_path = os.path.join(OUTPUT_DIR, f"{name}_anny.glb")
        mesh.export(obj_path)
        mesh.export(glb_path)
        print(f"    [OK] {obj_path}")
        print(f"    [OK] {glb_path}")
        print(f"         {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    print(f"\n[SUCCESS] Outputs saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
