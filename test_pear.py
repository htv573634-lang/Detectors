import os
import glob
import sys
import numpy as np
import torch
import trimesh
from PIL import Image

INPUT_DIR = "inputs"
OUTPUT_DIR = "out_pear"
MODEL_ASSETS = "PEAR/assets"
DEVICE = "cpu"

IMAGE_EXTS = ("jpg", "jpeg", "jpge", "png", "bmp", "webp", "tif", "tiff")

def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = find_images()
    if not images:
        print(f"[ERROR] No images in {INPUT_DIR}/")
        sys.exit(1)
    print(f"[*] Found {len(images)} image(s)")

    # ---- Load PEAR ----
    # ⚠️ YOU MUST REPLACE THIS SECTION with PEAR's actual loading code.
    # Look at PEAR's `inference_images.py` to find the correct class and function names.
    sys.path.insert(0, "PEAR")
    try:
        # Placeholder: This is a hypothetical class name.
        from pear.utils.inference import PEARInference
        model = PEARInference(model_path=MODEL_ASSETS, device=DEVICE)
        model.eval()
        print("[OK] PEAR loaded on CPU")
    except Exception as e:
        print(f"[ERROR] Failed to load PEAR: {e}")
        print("        -> Check PEAR's README and `inference_images.py` for the correct API.")
        sys.exit(1)

    # ---- Process Each Image ----
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[*] Processing {img_path}")
        try:
            img = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                # Placeholder: Replace with the actual prediction call.
                result = model.predict(img)
            
            verts = np.asarray(result["vertices"])
            faces = np.asarray(result["faces"])
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            out_obj = os.path.join(OUTPUT_DIR, f"{name}_pear.obj")
            mesh.export(out_obj)
            print(f"    [OK] Exported {out_obj}")
        except Exception as e:
            print(f"    [ERROR] {name}: {e}")

if __name__ == "__main__":
    main()
