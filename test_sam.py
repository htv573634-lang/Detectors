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

def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def snapshot_sam_dir():
    """Record existing files in SAM_DIR before running."""
    files = set()
    for f in glob.glob(os.path.join(SAM_DIR, "**", "*"), recursive=True):
        if os.path.isfile(f):
            files.add(f)
    return files

def collect_new_files(before, stem):
    """Copy any new file created in SAM_DIR into out_sam/<stem>_<filename>."""
    after = snapshot_sam_dir()
    new_files = after - before
    if not new_files:
        print("    [WARN] No new output files detected.")
        return
    for f in new_files:
        # Skip model files, source files, and build artifacts
        if any(skip in f for skip in ["/onnx/", "/build/", "/src/", "/include/", ".onnx", ".gguf", ".lbs", ".bin"]):
            continue
        # Only keep common output extensions
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".obj", ".bvh", ".json", ".ply", ".glb", ".gltf", ".fbx", ".npz", ".npy", ".txt", ".mtl"):
            continue
        base = os.path.basename(f)
        dest = os.path.join(OUTPUT_DIR, f"{stem}_{base}")
        shutil.copy2(f, dest)
        print(f"    Copied {f} -> {dest}")

def run_sam_on_image(image_path, image_name):
    stem = os.path.splitext(image_name)[0]
    rel_image_path = os.path.relpath(image_path, SAM_DIR)

    # Ensure binary is executable
    if not os.path.isfile(SAM_BINARY):
        print(f"    [ERROR] Binary not found: {SAM_BINARY}")
        # Try to find any executable in build/
        candidates = glob.glob(os.path.join(SAM_DIR, "build", "*"))
        candidates = [c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)]
        print(f"    Available executables in build/: {[os.path.basename(c) for c in candidates]}")
        return False

    cmd = [
        SAM_BINARY,
        "--from", rel_image_path,
        "--onnx-dir", ONNX_DIR,
        "--cpu"
    ]

    before = snapshot_sam_dir()
    print(f"[*] Running: {' '.join(cmd)} (cwd={SAM_DIR})")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SAM_DIR)

    # Always print stdout/stderr for debugging
    if result.stdout.strip():
        print(f"    [STDOUT]\n{result.stdout}")
    if result.stderr.strip():
        print(f"    [STDERR]\n{result.stderr}")

    if result.returncode != 0:
        print(f"    [ERROR] Exit code: {result.returncode}")
        return False

    print(f"    [OK] Exit code: 0")
    collect_new_files(before, stem)
    return True

def main():
    setup_dirs()

    if not os.path.isdir(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"[!] Created {INPUT_DIR}. Add images and rerun.")
        return

    # Debug: show what's in onnx/
    print(f"[*] ONNX directory: {ONNX_DIR}")
    if os.path.isdir(ONNX_DIR):
        print(f"    Contents: {os.listdir(ONNX_DIR)}")
    else:
        print(f"    [WARN] ONNX directory does not exist!")

    # Debug: show binary
    print(f"[*] Binary: {SAM_BINARY}")
    print(f"    Exists: {os.path.isfile(SAM_BINARY)}")

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
