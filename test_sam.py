import os
import subprocess
import glob
import shutil
import time

INPUT_DIR = os.path.abspath("inputs")
OUTPUT_DIR = os.path.abspath("out_sam")
SAM_DIR = os.path.abspath("SAM3DBody-cpp")
SAM_BINARY = os.path.join(SAM_DIR, "build", "fast_sam_3dbody_run")
ONNX_DIR = os.path.join(SAM_DIR, "onnx")

MESH_DIR = os.path.join(OUTPUT_DIR, "meshes")
BVH_DIR = os.path.join(OUTPUT_DIR, "bvh")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

def setup_dirs():
    for d in [MESH_DIR, BVH_DIR, JSON_DIR]:
        os.makedirs(d, exist_ok=True)

def run_sam_on_image(image_path, image_name):
    stem = os.path.splitext(image_name)[0]
    # Run from SAM_DIR so the binary finds its default folders
    rel_image_path = os.path.relpath(image_path, SAM_DIR)
    cmd = [
        SAM_BINARY,
        "--from", rel_image_path,
        "--onnx-dir", ONNX_DIR,
        "--cpu"
    ]
    print(f"[*] Running: {' '.join(cmd)} (cwd={SAM_DIR})")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SAM_DIR)
    if result.returncode != 0:
        print(f"    [ERROR] {result.stderr}")
        return False
    print(f"    [OK] {result.stdout.strip()}")

    # After run, find newly created files in SAM_DIR and move them
    now = time.time()
    for ext, dest_dir in [(".obj", MESH_DIR), (".bvh", BVH_DIR), (".json", JSON_DIR)]:
        for f in glob.glob(os.path.join(SAM_DIR, f"*{ext}")):
            if os.path.getmtime(f) > now - 60:  # created/modified in last 60 sec
                dest = os.path.join(dest_dir, f"{stem}{ext}")
                shutil.move(f, dest)
                print(f"    Moved {os.path.basename(f)} -> {dest}")
    return True

def main():
    setup_dirs()
    if not os.path.isdir(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"[!] Created {INPUT_DIR}. Add images and rerun.")
        return

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_ext)]

    if not images:
        print(f"[!] No images in {INPUT_DIR}.")
        return

    print(f"[*] Found {len(images)} image(s).")
    success, fail = 0, 0

    for i, img in enumerate(images, 1):
        print(f"\n--- [{i}/{len(images)}] {img} ---")
        path = os.path.join(INPUT_DIR, img)
        if run_sam_on_image(path, img):
            success += 1
        else:
            fail += 1

    print(f"\n[SUMMARY] Success: {success} | Failed: {fail}")
    print(f"[*] Outputs saved in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
