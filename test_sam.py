import os
import subprocess

INPUT_DIR = "inputs"
OUTPUT_DIR = "out_sam"
# Correct binary name and path
SAM_BINARY = "SAM3DBody-cpp/build/fast_sam_3dbody_run"

MESH_DIR = os.path.join(OUTPUT_DIR, "meshes")
BVH_DIR = os.path.join(OUTPUT_DIR, "bvh")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

def setup_dirs():
    for d in [MESH_DIR, BVH_DIR, JSON_DIR]:
        os.makedirs(d, exist_ok=True)

def run_sam_on_image(image_path, image_name):
    stem = os.path.splitext(image_name)[0]

    mesh_out = os.path.join(MESH_DIR, f"{stem}.obj")
    bvh_out  = os.path.join(BVH_DIR, f"{stem}.bvh")
    json_out = os.path.join(JSON_DIR, f"{stem}.json")

    # Correct command-line arguments based on the C++ CLI
    # The --from flag is the primary input, outputs are handled via config or defaults
    cmd = [
        SAM_BINARY,
        "--from", image_path,
        # Note: The C++ CLI may not support custom output paths directly.
        # It often saves outputs to a default folder.
        # If custom paths fail, the script below will still run,
        # but you may need to check the default output location.
    ]

    print(f"[*] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    [ERROR] {result.stderr}")
        return False

    print(f"    [OK] {result.stdout.strip()}")
    # Post-run: Move outputs to our desired folders if the CLI saved them elsewhere
    # This part depends on the actual behavior of the C++ binary.
    # For now, we assume it might save to a default location or we handle it later.
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
    print(f"[*] Check outputs in: {OUTPUT_DIR}/ and the default output folder of the C++ binary.")

if __name__ == "__main__":
    main()
