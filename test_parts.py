import os
import glob
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh

INPUT_DIR   = "inputs"
OUTPUT_DIR  = "artifacts"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"
Z_AMPLIFY   = 2.0
MAX_SIDE    = 700
CLOSURE     = "mirror"   # "mirror" or "offset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTS = ("jpg","jpeg","jpge","jpe","jfif","png","bmp","webp","tif","tiff")
files = []
for ext in IMAGE_EXTS:
    for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
        files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"[INFO] Using image: {IMAGE_PATH}")

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"[INFO] Image size: {W}x{H}")

# Stage 1: Segment foreground
print("[INFO] Stage 1: rembg foreground segmentation...")
from rembg import remove, new_session
session = new_session("isnet-general-use")
rgba = remove(Image.fromarray(img_rgb), session=session)
alpha = np.array(rgba)[..., 3]
mask = (alpha > 127).astype(np.uint8) * 255

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
if n > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ((lab == largest).astype(np.uint8)) * 255

mask = cv2.GaussianBlur(mask, (7, 7), 0)
mask = (mask > 127).astype(np.uint8) * 255

ys, xs = np.where(mask > 0)
if len(xs) == 0:
    raise RuntimeError("Empty mask")
x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
print(f"[OK] Subject bbox: ({x0},{y0}) -> ({x1},{y1})")

# Stage 2: Depth
print("[INFO] Stage 2: Depth Anything V2...")
depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
out = depth_pipe(Image.fromarray(img_rgb))
if isinstance(out, list): out = out[0]

if "predicted_depth" in out:
    raw = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(out["depth"]).astype(np.float32)

if raw.shape != (H, W):
    raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LANCZOS4)

raw_crop = raw[y0:y1, x0:x1]
mask_crop = mask[y0:y1, x0:x1]
rgb_crop = img_rgb[y0:y1, x0:x1].copy()
rgb_crop[mask_crop == 0] = 255

sv = raw_crop[mask_crop > 0]
lo, hi = np.percentile(sv, 2), np.percentile(sv, 98)
disp = np.clip((raw_crop - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

disp_u8 = (disp * 255).astype(np.uint8)
disp_u8 = cv2.medianBlur(disp_u8, 5)
disp = disp_u8.astype(np.float32) / 255.0

ch, cw = disp.shape
for _ in range(3):
    s = 96 / max(ch, cw)
    small = cv2.resize(disp, (max(8, int(cw*s)), max(8, int(ch*s))),
                       interpolation=cv2.INTER_AREA)
    disp = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_CUBIC)
    disp = np.clip(disp, 0, 1)

disp = cv2.bilateralFilter(disp, 9, 0.2, 15)
disp = np.clip(disp, 0, 1)
print(f"[OK] Depth range: {disp.min():.3f} -> {disp.max():.3f}, std={disp.std():.3f}")

# Save previews
cv2.imwrite(os.path.join(OUTPUT_DIR, f"parts_depth_{base_name}.png"),
            (disp * 255).astype(np.uint8))
cv2.imwrite(os.path.join(OUTPUT_DIR, f"parts_mask_{base_name}.png"), mask)

# Stage 3: Build mesh
print("[INFO] Stage 3: Building mesh...")
h, w = disp.shape
scale = min(1.0, MAX_SIDE / max(h, w))
if scale < 1.0:
    nw, nh = int(w*scale), int(h*scale)
    disp = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_CUBIC)
    rgb_crop = cv2.resize(rgb_crop, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_crop = cv2.resize(mask_crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    h, w = nh, nw
print(f"[INFO] Mesh grid: {w}x{h}")

z = (1.0 - disp) * Z_AMPLIFY

us, vs = np.meshgrid(np.arange(w), np.arange(h))
X = (us - w/2) / max(w, h)
Y = (vs - h/2) / max(w, h)

eps = 1.0 / max(w, h) / 4.0
uv_front = np.stack([us.ravel()/(w-1), 1.0 - vs.ravel()/(h-1)], axis=-1)
uv_front = np.clip(uv_front, eps, 1.0 - eps)

mb = (mask_crop > 127).astype(np.uint8)
md = cv2.dilate(mb, np.ones((3,3), np.uint8), 1)

m_tl = md[:-1, :-1].astype(bool); m_tr = md[:-1, 1:].astype(bool)
m_br = md[1:, 1:].astype(bool);   m_bl = md[1:, :-1].astype(bool)
vc = m_tl & m_tr & m_br & m_bl

idx = np.arange(h*w).reshape(h, w)
a = idx[:-1,:-1][vc]; b = idx[:-1,1:][vc]
c = idx[1:,1:][vc];   d = idx[1:,:-1][vc]
print(f"[INFO] Valid cells: {vc.sum()} / {(h-1)*(w-1)}")

front_faces = np.vstack([
    np.stack([a, b, c], axis=1),
    np.stack([a, c, d], axis=1),
])

zf = z.ravel()
z_mean = zf[mb.ravel() > 0].mean()
if CLOSURE == "mirror":
    z_back = (2*z_mean - zf).reshape(h, w)
else:
    z_back = (zf + 0.8).reshape(h, w)

fv = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)
bv = np.stack([X.ravel(), Y.ravel(), z_back.ravel()], axis=-1)
verts = np.vstack([fv, bv])

solid_uv = np.full_like(uv_front, 0.005)
uv = np.vstack([uv_front, solid_uv])

off = h*w
back_faces = np.vstack([
    np.stack([a+off, c+off, b+off], axis=1),
    np.stack([a+off, d+off, c+off], axis=1),
])

V = vc
above = np.zeros_like(V); above[1:,:] = V[:-1,:]
below = np.zeros_like(V); below[:-1,:] = V[1:,:]
left  = np.zeros_like(V); left[:,1:] = V[:,:-1]
right = np.zeros_like(V); right[:,:-1] = V[:,1:]

seams = []
for boundary, corners in [
    (V & ~above,  "top"),
    (V & ~below,  "bottom"),
    (V & ~left,   "left"),
    (V & ~right,  "right"),
]:
    rows, cols = np.where(boundary)
    for r, col in zip(rows, cols):
        if corners == "top":
            f0, f1 = r*w + col, r*w + col + 1
            seams.append([f0, f0+off, f1+off]); seams.append([f0, f1+off, f1])
        elif corners == "bottom":
            f0, f1 = (r+1)*w + col, (r+1)*w + col + 1
            seams.append([f0, f1, f1+off]); seams.append([f0, f1+off, f0+off])
        elif corners == "left":
            f0, f1 = r*w + col, (r+1)*w + col
            seams.append([f0, f1, f1+off]); seams.append([f0, f1+off, f0+off])
        else:
            f0, f1 = r*w + col + 1, (r+1)*w + col + 1
            seams.append([f0, f0+off, f1+off]); seams.append([f0, f1+off, f1])

seams = np.array(seams, dtype=np.int64)
faces = np.vstack([front_faces, back_faces, seams])
print(f"[INFO] Mesh: {len(verts)} verts, {len(faces)} faces")

mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()

try:
    trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5, iterations=4)
    print("[OK] Humphrey smoothing applied")
except Exception as e:
    print(f"[WARN] Smoothing skipped: {e}")

texture_img = Image.fromarray(rgb_crop)
mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=texture_img)

# Stage 4: Export
print("[INFO] Stage 4: Exporting...")
mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
mesh.faces = np.asarray(mesh.faces, dtype=np.int64)

glb = os.path.join(OUTPUT_DIR, f"parts_{base_name}.glb")
mesh.export(glb, file_type="glb")
print(f"[OK] GLB: {glb} ({os.path.getsize(glb)/1024:.1f} KB)")

obj = os.path.join(OUTPUT_DIR, f"parts_{base_name}.obj")
mesh.export(obj, file_type="obj")
print(f"[OK] OBJ: {obj} ({os.path.getsize(obj)/1024:.1f} KB)")

print("\n=== Contents of artifacts/ ===")
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p)/1024:.1f} KB)")

print("\nDONE.")
