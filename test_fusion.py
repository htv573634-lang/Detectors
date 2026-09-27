import os
import glob
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh

# ── Config ────────────────────────────────────────────────────
INPUT_DIR       = "inputs2"
HMR_DIR         = "out_hmr2"
OUTPUT_DIR      = "out_fusion"
DEPTH_MODEL     = "depth-anything/Depth-Anything-V2-Base-hf"
DETAIL_STRENGTH = 0.6      # how strongly detail displaces surface (0.1–1.5)
BLUR_SIGMA      = 25       # smoothness of low-freq body shape (20–50)
SMOOTH_ITERS    = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Find HMR2.0 params ────────────────────────────────────────
params_files = sorted(
    glob.glob(os.path.join(HMR_DIR, "hmr2_params_*.npz")),
    key=os.path.getmtime, reverse=True
)
if not params_files:
    raise FileNotFoundError(
        f"No HMR2.0 params in {HMR_DIR}/ — run test_hmr2.yml first"
    )
params_path = params_files[0]
print(f"[INFO] Using params: {params_path}")

data = np.load(params_path, allow_pickle=True)
vertices   = data["vertices"]
faces      = data["faces"]
bbox       = data["detection_box"]
img_h      = int(data["image_h"])
img_w      = int(data["image_w"])
print(f"[INFO] Mesh: {len(vertices)} verts, {len(faces)} faces")
print(f"[INFO] Bbox: {bbox.tolist()}")
print(f"[INFO] Image: {img_w}x{img_h}")

# ── Find original image ───────────────────────────────────────
IMAGE_EXTS = ("jpg","jpeg","jpge","png","bmp","webp","tif","tiff")
files = []
for ext in IMAGE_EXTS:
    for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
        files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"[INFO] Image: {IMAGE_PATH}")

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── Stage 1: Depth Anything V2 ───────────────────────────────
print("[INFO] Stage 1: Running Depth Anything V2...")
depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
out = depth_pipe(Image.fromarray(img_rgb))
if isinstance(out, list):
    out = out[0]

if "predicted_depth" in out:
    raw = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(out["depth"]).astype(np.float32)

if raw.shape != (img_h, img_w):
    raw = cv2.resize(raw, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)

# Normalize to 0..1 (1 = near camera)
d = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

# Save full depth preview
cv2.imwrite(os.path.join(OUTPUT_DIR, f"fusion_depth_{base_name}.png"),
            (d * 255).astype(np.uint8))
print(f"[OK] Depth map saved")

# ── Stage 2: Extract high-frequency detail ───────────────────
print("[INFO] Stage 2: Extracting high-frequency detail...")
d_blur = cv2.GaussianBlur(d, (0, 0), BLUR_SIGMA)
detail = d - d_blur

# Normalize detail to [-1, 1]
max_abs = np.abs(detail).max() + 1e-8
detail = detail / max_abs
print(f"[INFO] Detail range: {detail.min():.3f} -> {detail.max():.3f}")

# Save detail preview
detail_vis = ((detail + 1.0) * 127.5).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"fusion_detail_{base_name}.png"),
            detail_vis)

# ── Stage 3: Load mesh and compute normals ───────────────────
print("[INFO] Stage 3: Loading HMR2.0 mesh...")
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()

print(f"[INFO] Loaded mesh: {len(mesh.vertices)} verts")
print(f"[INFO] Normals shape: {mesh.vertex_normals.shape}")

# ── Stage 4: Project vertices to pixel coordinates ───────────
print("[INFO] Stage 4: Projecting vertices to image...")
verts = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)

vx, vy = verts[:, 0], verts[:, 1]
x_min, x_max = vx.min(), vx.max()
y_min, y_max = vy.min(), vy.max()

# Normalize to [0, 1]; flip Y (image Y goes down, mesh Y goes up)
nx = (vx - x_min) / (x_max - x_min + 1e-8)
ny = 1.0 - (vy - y_min) / (y_max - y_min + 1e-8)

# Map to bounding box pixel coordinates
bx1, by1, bx2, by2 = bbox
px = (bx1 + nx * (bx2 - bx1)).astype(int)
py = (by1 + ny * (by2 - by1)).astype(int)

px = np.clip(px, 0, img_w - 1)
py = np.clip(py, 0, img_h - 1)
print(f"[INFO] Pixel range: x=[{px.min()}, {px.max()}], y=[{py.min()}, {py.max()}]")

# ── Stage 5: Sample detail at each vertex ────────────────────
print("[INFO] Stage 5: Sampling detail at vertices...")
vertex_detail = detail[py, px]

# Only apply to front-facing vertices (normal Z > 0 toward camera)
front_mask = normals[:, 2] > 0.1
print(f"[INFO] Front-facing vertices: {front_mask.sum()} / {len(verts)}")

# ── Stage 6: Displace vertices along normals ─────────────────
print("[INFO] Stage 6: Displacing vertices...")
displacement = np.zeros_like(verts)
displacement[front_mask] = (
    normals[front_mask] *
    (vertex_detail[front_mask, None] * DETAIL_STRENGTH)
)

new_verts = verts + displacement
mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)

# ── Stage 7: Repair mesh ─────────────────────────────────────
print("[INFO] Stage 7: Repairing mesh...")
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()
trimesh.repair.fix_normals(mesh)
trimesh.repair.fix_inversion(mesh)
trimesh.repair.fill_holes(mesh)

print(f"[INFO] Watertight: {mesh.is_watertight}")
print(f"[INFO] Winding consistent: {mesh.is_winding_consistent}")

try:
    trimesh.smoothing.filter_humphrey(
        mesh, alpha=0.05, beta=0.3, iterations=SMOOTH_ITERS
    )
    print("[OK] Humphrey smoothing applied")
except Exception as e:
    print(f"[WARN] Smoothing skipped: {e}")

# Re-repair after smoothing
trimesh.repair.fix_normals(mesh)
trimesh.repair.fill_holes(mesh)

# ── Stage 8: Export ──────────────────────────────────────────
print("[INFO] Stage 8: Exporting...")
mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
mesh.faces = np.asarray(mesh.faces, dtype=np.int64)

glb_path = os.path.join(OUTPUT_DIR, f"fusion_{base_name}.glb")
mesh.export(glb_path, file_type="glb")
print(f"[OK] GLB: {glb_path} ({os.path.getsize(glb_path)/1024:.1f} KB)")

obj_path = os.path.join(OUTPUT_DIR, f"fusion_{base_name}.obj")
mesh.export(obj_path, file_type="obj")
print(f"[OK] OBJ: {obj_path} ({os.path.getsize(obj_path)/1024:.1f} KB)")

print(f"\n=== Contents of {OUTPUT_DIR}/ ===")
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p)/1024:.1f} KB)")

print("\nDONE.")
