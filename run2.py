import os
import glob
import shutil
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh

# ── Config ────────────────────────────────────────────────────
INPUT_DIR         = "inputs2"
OUTPUT_DIR        = "artifacts2"
DEPTH_MODEL       = "depth-anything/Depth-Anything-V2-Base-hf"
Z_AMPLIFY         = 3.0
BACK_THICKNESS    = 1.0
MAX_SIDE          = 500
MIRROR_MODE       = True
SMOOTH_ITERS      = 6
DOWNSAMPLE_PASSES = 2
DOWNSAMPLE_SIZE   = 96
MASK_ERODE_PX     = 3
MAX_TEXTURE_SIZE  = 512

UV_CLAMP         = True
UV_BACK_SOLID    = True
UV_DEBUG         = True

# ── Cleanup artifacts2/ ──────────────────────────────────────
if os.path.isdir(OUTPUT_DIR):
    for f in os.listdir(OUTPUT_DIR):
        if f == ".gitkeep":
            continue
        path = os.path.join(OUTPUT_DIR, f)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"⚠️  Could not delete {path}: {e}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Cleared {OUTPUT_DIR}/ (kept .gitkeep)")

# ── Universal image finder ────────────────────────────────────
IMAGE_EXTS = (
    "jpg", "jpeg", "jpge", "jpe", "jfif", "jif", "jfi",
    "png", "bmp", "webp", "tif", "tiff",
    "heic", "heif", "avif", "gif",
)
files = []
for ext in IMAGE_EXTS:
    for pattern in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
        files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))

files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
if not files:
    raise FileNotFoundError(f"No image found in {INPUT_DIR}/")

files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

# ── Load image ────────────────────────────────────────────────
img_pil = Image.open(IMAGE_PATH)
has_alpha = img_pil.mode in ("RGBA", "LA") or (
    img_pil.mode == "P" and "transparency" in img_pil.info
)
print(f"Image mode: {img_pil.mode}, has_alpha: {has_alpha}")

img_pil_rgb = img_pil.convert("RGB")
img_rgb = np.array(img_pil_rgb)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
H, W = img_rgb.shape[:2]

alpha_channel = None
if has_alpha:
    alpha_channel = np.array(img_pil.convert("RGBA"))[..., 3]
    print(f"Alpha range: {alpha_channel.min()} → {alpha_channel.max()}")

# ── Stage 1: Detect body part ────────────────────────────────
print("\n── STAGE 1: Detect body part ──")

mask = np.zeros((H, W), dtype=np.uint8)
labels = []
mask_source = "none"

if alpha_channel is not None and alpha_channel.min() < 200 and alpha_channel.max() > 50:
    mask = (alpha_channel > 127).astype(np.uint8) * 255
    mask_source = "alpha-channel"
    labels = ["subject"]
    print("Using alpha channel as mask")

if mask_source == "none":
    try:
        from rembg import remove, new_session
        session = new_session("u2netp")
        rgba = remove(img_pil_rgb, session=session)
        mask = (np.array(rgba)[..., 3] > 127).astype(np.uint8) * 255
        mask_source = "rembg"
        labels = ["subject"]
        print("rembg mask generated")
    except Exception as e:
        print(f"rembg unavailable ({e})")

if mask_source == "none":
    mask[:] = 255
    mask_source = "full-image"
    labels = ["full"]
    print("Using full image")

print(f"Mask source: {mask_source}")

# ── Mask cleanup ─────────────────────────────────────────────
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

num_labels, lab_imgs, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ((lab_imgs == largest).astype(np.uint8)) * 255

mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=MASK_ERODE_PX)
mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=MASK_ERODE_PX - 1)
mask = cv2.GaussianBlur(mask, (7, 7), 0)
mask = (mask > 127).astype(np.uint8) * 255

ys, xs = np.where(mask > 0)
if len(xs) == 0:
    raise RuntimeError("Mask is empty")
x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
print(f"Mask bbox: ({x0},{y0}) → ({x1},{y1})")
print(f"Mask coverage: {100*mask.mean()/255:.1f}% of image")

cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{base_name}.png"), mask)

# ── Stage 2: Depth Anything ──────────────────────────────────
print("\n── STAGE 2: Depth Anything ──")

depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
out = depth_pipe(img_pil_rgb)
if isinstance(out, list): out = out[0]

if "predicted_depth" in out:
    raw = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(out["depth"]).astype(np.float32)

if raw.shape != (H, W):
    raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LANCZOS4)

raw_crop  = raw[y0:y1, x0:x1]
mask_crop = mask[y0:y1, x0:x1]
rgb_crop  = img_rgb[y0:y1, x0:x1].copy()
rgb_crop[mask_crop == 0] = 255

subject_vals = raw_crop[mask_crop > 0]
if len(subject_vals) < 10:
    subject_vals = raw_crop.flatten()
lo, hi = np.percentile(subject_vals, 2), np.percentile(subject_vals, 98)
disp = np.clip((raw_crop - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

print(f"Applying {DOWNSAMPLE_PASSES}× downsample-upsample...")
ch, cw = disp.shape
for i in range(DOWNSAMPLE_PASSES):
    scale_small = DOWNSAMPLE_SIZE / max(ch, cw)
    small_h = max(8, int(ch * scale_small))
    small_w = max(8, int(cw * scale_small))
    small = cv2.resize(disp, (small_w, small_h), interpolation=cv2.INTER_AREA)
    disp = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_CUBIC)
    disp = np.clip(disp, 0.0, 1.0)
    disp = cv2.GaussianBlur(disp, (3, 3), 0)
    disp = np.clip(disp, 0.0, 1.0)

disp = cv2.bilateralFilter(disp, d=9, sigmaColor=0.20, sigmaSpace=15)
disp = np.clip(disp, 0.0, 1.0)

print(f"Depth range: {disp.min():.3f} → {disp.max():.3f}, std={disp.std():.3f}")

# ── Stage 3: Build mesh ──────────────────────────────────────
print("\n── STAGE 3: Reconstruct mesh ──")

h, w = disp.shape
scale = min(1.0, MAX_SIDE / max(h, w))
if scale < 1.0:
    new_w, new_h = int(w * scale), int(h * scale)
    disp      = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    rgb_crop  = cv2.resize(rgb_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_crop = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    h, w = new_h, new_w
print(f"Mesh grid: {w}×{h}")

z = (1.0 - disp) * Z_AMPLIFY

us, vs = np.meshgrid(np.arange(w), np.arange(h))
X = (us - w / 2) / max(w, h)
Y = (vs - h / 2) / max(w, h)

eps = 1.0 / max(w, h) / 4.0
uv_front = np.stack([us.ravel() / (w - 1), 1.0 - vs.ravel() / (h - 1)], axis=-1)
if UV_CLAMP:
    uv_front = np.clip(uv_front, eps, 1.0 - eps)

mask_bin = (mask_crop > 127).astype(np.uint8)
mask_dil = cv2.dilate(mask_bin, np.ones((3, 3), np.uint8), iterations=1)

m_tl = mask_dil[:-1, :-1].astype(bool)
m_tr = mask_dil[:-1, 1:].astype(bool)
m_br = mask_dil[1:, 1:].astype(bool)
m_bl = mask_dil[1:, :-1].astype(bool)
valid_cells = m_tl & m_tr & m_br & m_bl

idx = np.arange(h * w).reshape(h, w)
a = idx[:-1, :-1][valid_cells]
b = idx[:-1, 1:][valid_cells]
c = idx[1:, 1:][valid_cells]
d = idx[1:, :-1][valid_cells]

print(f"Valid grid cells: {valid_cells.sum()} / {(h-1)*(w-1)}")

if len(a) == 0:
    raise RuntimeError("No valid mesh cells")

front_faces = np.vstack([
    np.stack([a, b, c], axis=1),
    np.stack([a, c, d], axis=1),
])

z_flat = z.ravel()
z_mean = z_flat[mask_bin.ravel() > 0].mean()
z_back = (z_flat + BACK_THICKNESS - 0.3 * (z_flat - z_mean)).reshape(h, w)
z_back = np.clip(z_back, z.min(), z.max() + BACK_THICKNESS + 0.5)

front_verts = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)
back_verts  = np.stack([X.ravel(), Y.ravel(), z_back.ravel()], axis=-1)
vertices = np.vstack([front_verts, back_verts])

if UV_BACK_SOLID:
    solid_uv = np.full_like(uv_front, 0.005)
    uv = np.vstack([uv_front, solid_uv])
else:
    uv = np.vstack([uv_front, uv_front])

off = h * w
back_faces = np.vstack([
    np.stack([a + off, c + off, b + off], axis=1),
    np.stack([a + off, d + off, c + off], axis=1),
])

V = valid_cells
above_valid = np.zeros_like(V); above_valid[1:, :] = V[:-1, :]
below_valid = np.zeros_like(V); below_valid[:-1, :] = V[1:, :]
left_valid  = np.zeros_like(V); left_valid[:, 1:] = V[:, :-1]
right_valid = np.zeros_like(V); right_valid[:, :-1] = V[:, 1:]

top_b    = V & (~above_valid)
bottom_b = V & (~below_valid)
left_b   = V & (~left_valid)
right_b  = V & (~right_valid)

seam_faces_list = []

rows, cols = np.where(top_b)
if len(rows):
    tl = rows * w + cols; tr = tl + 1
    seam_faces_list.append(np.stack([tl, tr, tr + off], axis=1))
    seam_faces_list.append(np.stack([tl, tr + off, tl + off], axis=1))

rows, cols = np.where(bottom_b)
if len(rows):
    bl = (rows + 1) * w + cols; br = bl + 1
    seam_faces_list.append(np.stack([bl, br, br + off], axis=1))
    seam_faces_list.append(np.stack([bl, br + off, bl + off], axis=1))

rows, cols = np.where(left_b)
if len(rows):
    tl = rows * w + cols; bl = tl + w
    seam_faces_list.append(np.stack([tl, bl, bl + off], axis=1))
    seam_faces_list.append(np.stack([tl, bl + off, tl + off], axis=1))

rows, cols = np.where(right_b)
if len(rows):
    tr = rows * w + (cols + 1); br = tr + w
    seam_faces_list.append(np.stack([tr, br, br + off], axis=1))
    seam_faces_list.append(np.stack([tr, br + off, tr + off], axis=1))

if seam_faces_list:
    seam_faces = np.vstack(seam_faces_list).astype(np.int64)
    faces = np.vstack([front_faces, back_faces, seam_faces])
else:
    faces = np.vstack([front_faces, back_faces])

print(f"Mesh raw: {len(vertices)} vertices, {len(faces)} faces")

# ── Build trimesh ────────────────────────────────────────────
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()
print(f"Pre-smooth: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

# ── FIX: Use HUMPHREY smoothing (stable on any mesh) ─────────
if len(mesh.faces) > 0:
    print(f"Applying {SMOOTH_ITERS} Humphrey Laplacian iterations...")
    try:
        trimesh.smoothing.filter_humphrey(
            mesh, alpha=0.1, beta=0.5, iterations=SMOOTH_ITERS
        )
        print("Humphrey smoothing applied")
    except Exception as e:
        print(f"Smoothing skipped: {e}")

mesh.remove_unreferenced_vertices()

# ── FINAL SAFETY: only remove NaN if SOME are NaN ────────────
verts_np = np.asarray(mesh.vertices)
faces_np = np.asarray(mesh.faces)

if len(verts_np) > 0:
    finite_verts = np.isfinite(verts_np).all(axis=1)
    n_bad = (~finite_verts).sum()
    if n_bad > 0 and n_bad < len(verts_np) * 0.5:
        # Only clean if <50% are NaN — otherwise keep original
        print(f"⚠️  Removing {n_bad} NaN/inf vertices ({100*n_bad/len(verts_np):.1f}%)")
        face_finite = finite_verts[faces_np].all(axis=1)
        faces_np = faces_np[face_finite]
        if len(faces_np) > 0:
            used = np.unique(faces_np)
            remap = -np.ones(len(verts_np), dtype=np.int64)
            remap[used] = np.arange(len(used))
            verts_np = verts_np[used]
            faces_np = remap[faces_np]
            mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    elif n_bad >= len(verts_np) * 0.5:
        print(f"⚠️  {n_bad}/{len(verts_np)} NaN — keeping original pre-smooth mesh")

print(f"After cleanup: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

# ── SANITY CHECK ─────────────────────────────────────────────
if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
    print("❌ ERROR: Mesh is empty after cleanup!")
    print("   Writing debug info to artifacts2/debug.txt")
    with open(os.path.join(OUTPUT_DIR, "debug.txt"), "w") as f:
        f.write(f"vertices pre: {len(vertices)}\n")
        f.write(f"faces pre: {len(faces)}\n")
        f.write(f"vertices post: {len(mesh.vertices)}\n")
        f.write(f"faces post: {len(mesh.faces)}\n")
    raise RuntimeError("Mesh became empty during processing")

# ── Stage 4: Texture ─────────────────────────────────────────
print("\n── STAGE 4: Texture ──")

tex_h, tex_w = rgb_crop.shape[:2]
if max(tex_h, tex_w) > MAX_TEXTURE_SIZE:
    tex_scale = MAX_TEXTURE_SIZE / max(tex_h, tex_w)
    new_tex_w = int(tex_w * tex_scale)
    new_tex_h = int(tex_h * tex_scale)
    rgb_crop = cv2.resize(rgb_crop, (new_tex_w, new_tex_h),
                          interpolation=cv2.INTER_AREA)
    print(f"Texture resized to {new_tex_w}×{new_tex_h}")

if rgb_crop.dtype != np.uint8:
    rgb_crop = rgb_crop.astype(np.uint8)

BORDER = 2
texture_padded = cv2.copyMakeBorder(
    rgb_crop, BORDER, BORDER, BORDER, BORDER,
    cv2.BORDER_REPLICATE
)
print(f"Texture padded to {texture_padded.shape[1]}×{texture_padded.shape[0]}")

pad_w = texture_padded.shape[1]
pad_h = texture_padded.shape[0]
uv_pad = uv.copy()
uv_pad[:, 0] = (uv[:, 0] * (pad_w - 2 * BORDER) + BORDER) / pad_w
uv_pad[:, 1] = (uv[:, 1] * (pad_h - 2 * BORDER) + BORDER) / pad_h
uv_pad = np.clip(uv_pad, 0.0, 1.0).astype(np.float32)

texture_path = os.path.join(OUTPUT_DIR, f"texture_{base_name}.png")
cv2.imwrite(texture_path, cv2.cvtColor(texture_padded, cv2.COLOR_RGB2BGR))
print(f"Texture saved: {texture_path}")

_, tex_bytes = cv2.imencode(".png", cv2.cvtColor(texture_padded, cv2.COLOR_RGB2BGR))
texture_bytes = tex_bytes.tobytes()

from trimesh.visual.material import PBRMaterial

material = PBRMaterial(
    baseColorTexture=texture_bytes,
    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
    metallicFactor=0.0,
    roughnessFactor=0.9,
    doubleSided=False,
)
visual = trimesh.visual.TextureVisuals(uv=uv_pad, material=material)
mesh.visual = visual
print("PBR material assigned")

# ── UV debug ─────────────────────────────────────────────────
if UV_DEBUG:
    debug = np.full((pad_h, pad_w, 3), 40, dtype=np.uint8)
    px = np.clip((uv_pad[:, 0] * (pad_w - 1)).astype(int), 0, pad_w - 1)
    py = np.clip(((1.0 - uv_pad[:, 1]) * (pad_h - 1)).astype(int), 0, pad_h - 1)
    n_front = len(uv_front)
    for i in range(0, n_front, 20):
        cv2.circle(debug, (px[i], py[i]), 1, (255, 255, 0), -1)
    for i in range(n_front, len(px), 50):
        cv2.circle(debug, (px[i], py[i]), 1, (0, 140, 255), -1)
    cv2.rectangle(debug, (BORDER, BORDER),
                  (pad_w - BORDER - 1, pad_h - BORDER - 1),
                  (255, 255, 255), 1)
    uv_debug_path = os.path.join(OUTPUT_DIR, f"uv_debug_{base_name}.png")
    cv2.imwrite(uv_debug_path, debug)
    print(f"UV debug saved: {uv_debug_path}")

# ── Stage 5: Export (GLB + OBJ + PLY for debugging) ──────────
print("\n── STAGE 5: Export ──")

mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
mesh.faces    = np.asarray(mesh.faces,    dtype=np.int64)
if mesh.visual.uv is not None:
    mesh.visual.uv = np.asarray(mesh.visual.uv, dtype=np.float32)

# Sanity check before export
print(f"Final mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
print(f"Bounding box: {mesh.bounds.tolist() if mesh.bounds is not None else 'none'}")

label_str = "+".join(labels)
suffix = "closed" if MIRROR_MODE else "open"

# --- GLB ---
glb_path = os.path.join(OUTPUT_DIR, f"{label_str}_{base_name}_{suffix}.glb")
try:
    mesh.export(glb_path, file_type='glb')
    glb_size = os.path.getsize(glb_path)
    print(f"✓ GLB: {glb_path} ({glb_size/1024:.1f} KB)")
    if glb_size < 5000:
        print(f"⚠️  GLB is suspiciously small ({glb_size} bytes)")
except Exception as e:
    print(f"❌ GLB export failed: {e}")

# --- OBJ (text-based, easy to debug) ---
obj_path = os.path.join(OUTPUT_DIR, f"{label_str}_{base_name}_{suffix}.obj")
try:
    mesh.export(obj_path, file_type='obj')
    obj_size = os.path.getsize(obj_path)
    print(f"✓ OBJ: {obj_path} ({obj_size/1024:.1f} KB)")
except Exception as e:
    print(f"❌ OBJ export failed: {e}")

# --- PLY (simple binary, very reliable) ---
ply_path = os.path.join(OUTPUT_DIR, f"{label_str}_{base_name}_{suffix}.ply")
try:
    mesh.export(ply_path, file_type='ply')
    ply_size = os.path.getsize(ply_path)
    print(f"✓ PLY: {ply_path} ({ply_size/1024:.1f} KB)")
except Exception as e:
    print(f"❌ PLY export failed: {e}")

# ── Verification ─────────────────────────────────────────────
print("\n── Verification ──")

for path in [glb_path, obj_path, ply_path]:
    if not os.path.exists(path):
        continue
    try:
        reloaded = trimesh.load(path, force='mesh')
        print(f"✓ {os.path.basename(path)}: "
              f"{len(reloaded.vertices)} verts, {len(reloaded.faces)} faces")
    except Exception as e:
        print(f"⚠️  {os.path.basename(path)}: reload failed: {e}")

print("\n── Contents of artifacts2/ ──")
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p)/1024:.1f} KB)")

print("\nDONE.")
