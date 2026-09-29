import os
import subprocess
import glob

INPUT_DIR = os.path.abspath("inputs")
OUTPUT_DIR = os.path.abspath("out_sam")
SAM_DIR = os.path.abspath("SAM3DBody-cpp")
SAM_BINARY = os.path.join(SAM_DIR, "build", "fast_sam_3dbody_run")
ONNX_DIR = os.path.join(SAM_DIR, "onnx")

def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # CRITICAL: Create the directory where the C++ patch will write mesh.vertices
    os.makedirs(os.path.join(SAM_DIR, "out_sam"), exist_ok=True)

def run_sam_on_image(image_path, image_name):
    stem = os.path.splitext(image_name)[0]
    rel_image_path = os.path.relpath(image_path, SAM_DIR)

    bvh_out = os.path.join(OUTPUT_DIR, f"{stem}.bvh")
    csv_out = os.path.join(OUTPUT_DIR, f"{stem}_keypoints.csv")

    cmd = [
        SAM_BINARY,
        "--from", rel_image_path,
        "--onnx-dir", ONNX_DIR,
        "--backbone", "backbone_fp32.onnx",
        "--cuda", "-1",
        "--bvh", bvh_out,
        "-o", csv_out,
        "--headless"
    ]

    print(f"[*] Running: {' '.join(cmd)} (cwd={SAM_DIR})")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SAM_DIR)

    if result.returncode != 0:
        print(f"    [ERROR] Exit code: {result.returncode}")
        print(f"    [STDERR]\n{result.stderr}")
        return False

    print(f"    [OK] Exit code: 0")

    # The C++ patch writes mesh.vertices inside SAM_DIR/out_sam/
    patched_verts = os.path.join(SAM_DIR, "out_sam", "mesh.vertices")
    if os.path.isfile(patched_verts):
        dest = os.path.join(OUTPUT_DIR, "mesh.vertices")
        os.replace(patched_verts, dest)
        print(f"    Moved mesh.vertices -> {dest}")
    else:
        direct = os.path.join(SAM_DIR, "mesh.vertices")
        if os.path.isfile(direct):
            dest = os.path.join(OUTPUT_DIR, "mesh.vertices")
            os.replace(direct, dest)
            print(f"    Moved mesh.vertices -> {dest}")
        else:
            print(f"    [WARN] mesh.vertices not found.")

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
