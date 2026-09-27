import os
import glob
import shutil
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh

# ── Config ────────────────────────────────────────────────────
INPUT_DIR        = "inputs2"
OUTPUT_DIR       = "artifacts2"
DEPTH_MODEL      = "depth-anything/Depth-Anything-V2-Base-hf"
Z_AMPLIFY        = 4.0          # raised for stronger curves
BACK_THICKNESS   = 1.5
MAX_SIDE         = 700
MIRROR_MODE      = True
SMOOTH_ITERS     = 8            # Laplacian smoothing iterations

# ── Cleanup artifacts2/ (keep .gitkeep) ──────────────────────
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
print(f"Base name  : {base_name}")

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

# ── Stage 1: Detect body part ────────────────────────────────
print("\n── STAGE 1: Detect body part ──")

mask = np.zeros((H, W), dtype=np.uint8)
labels = []
mask_source = "none"

# Priority 1: alpha channel
if has_alpha:
    alpha = np.array(img_pil.convert("RGBA"))[..., 3]
    if alpha.min() < 200 and alpha.max() > 50:
        mask = (alpha > 127).astype(np.uint8) * 255
        mask_source = "alpha-channel"
        labels.append("subject")
        print("Using embedded alpha channel as mask")

# Priority 2: Sapiens
if mask_source == "none":
    try:
        import torch
        from sapiens_inference import (
            SapiensPredictor, SapiensConfig, SapiensSegmentationType,
        )
        from sapiens_inference.normal import SapiensNormalType
        from sapiens_inference.depth import SapiensDepthType

        cfg = SapiensConfig()
        cfg.device = torch.device("cpu")
        cfg.segmentation_type = SapiensSegmentationType.SEGMENTATION_03B
        cfg.depth_type  = SapiensDepthType.OFF
        cfg.normal_type = SapiensNormalType.OFF

        predictor = SapiensPredictor(cfg)
        seg_result = predictor(img_rgb)

        if isinstance(seg_result, dict):
            seg_mask_ids = seg_result.get("segmentation",
                            seg_result.get("mask",
                            seg_result.get("seg_map")))
        else:
            seg_mask_ids = seg_result

        seg_mask_ids = np.asarray(seg_mask_ids).astype(np.uint8)
        mask = (seg_mask_ids > 0).astype(np.uint8) * 255
        mask_source = "sapiens"

        uc = np.unique(seg_mask_ids)
        if 2 in uc:  labels.append("face")
        if 5 in uc:  labels.append("left_hand")
        if 14 in uc: labels.append("right_hand")
        if 1 in uc or 21 in uc or 22 in uc: labels.append("body")
        if not labels: labels = ["full"]
        print(f"Sapiens parts: {'+'.join(labels)}")
    except Exception as e:
        print(f"Sapiens unavailable ({type(e).__name__}: {e})")
        print("→ Trying rembg fallback.")

# Priority 3: rembg
if mask_source == "none":
    try:
        from rembg import remove, new_session
        session = new_session("u2netp")
        rgba = remove(img_pil_rgb, session=session)
        mask = (np.array(rgba)[..., 3] > 127).astype(np.uint8) * 255
        mask_source = "rembg"
        labels.append("subject")
        print("rembg mask generated")
    except Exception as e:
        print(f"rembg unavailable ({e})")

# Priority 4: MediaPipe
if mask_source == "none":
    try:
        import mediapipe as mp
        with mp.solutions.holistic.Holistic(
                static_image_mode=True, model_complexity=2,
                min_detection_confidence=0.5) as holistic:
            result = holistic.process(img_rgb)

        def draw_mask(lms, mask, W, H, thickness):
            pts = np.array([[int(lm.x * W), int(lm.y * H)] for lm in lms.landmark])
            if len(pts) >= 3:
                cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
                for p in pts:
                    cv2.circle(mask, tuple(p), thickness, 255, -1)

        if result.face_landmarks:
            draw_mask(result.face_landmarks, mask, W, H, 10); labels.append("face")
        if result.left_hand_landmarks:
            draw_mask(result.left_hand_landmarks, mask, W, H, 20); labels.append("left_hand")
        if result.right_hand_landmarks:
            draw_mask(result.right_hand_landmarks, mask, W, H, 20); labels.append("right_hand")
        if result.pose_landmarks:
            draw_mask(result.pose_landmarks, mask, W, H, 40); labels.append("body")

        if not labels:
            mask[:] = 255; labels = ["full"]
        mask_source = "mediapipe"
        print(f"MediaPipe parts: {'+'.join(labels)}")
    except Exception as e:
        print(f"MediaPipe unavailable ({e})")
        mask[:] = 255; labels = ["full"]; mask_source = "full-image"

print(f"Mask source: {mask_source}")

# ── Mask cleanup ─────────────────────────────────────────────
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=1)

num_labels, lab_imgs, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ((lab_imgs == largest).astype(np.uint8)) * 255

mask = cv2.GaussianBlur(mask, (5, 5), 0)
mask = (mask > 127).astype(np.uint8) * 255

ys, xs = np.where(mask > 0)
if len(xs) == 0:
    raise RuntimeError("Mask is empty")
x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1

print(f"Mask coverage: {100*mask.mean()/255:.1f}% of image")
print(f"Subject bbox: ({x0},{y0}) → ({x1},{y1})")
print(f"Mask coverage in bbox: {100*mask[y0:y1, x0:x1].mean()/255:.1f}%")

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

# Normalize subject depth to 0..1
subject_vals = raw_crop[mask_crop > 0]
if len(subject_vals) < 10:
    subject_vals = raw_crop.flatten()
lo, hi = np.percentile(subject_vals, 2), np.percentile(subject_vals, 98)
disp = np.clip((raw_crop - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

# ── FIX 1: Bilateral filter (removes striping, keeps edges) ─
disp = cv2.bilateralFilter(disp, d=9, sigmaColor=0.15, sigmaSpace=15)

# ── FIX 2: CLAHE local contrast enhancement ──────────────────
disp_u8 = (disp * 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
disp_u8 = clahe.apply(disp_u8)
disp = disp_u8.astype(np.float32) / 255.0

# Second bilateral pass to seal in the enhanced curves
disp = cv2.bilateralFilter(disp, d=7, sigmaColor=0.10, sigmaSpace=10)

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

uv = np.stack([us.ravel() / (w - 1), 1.0 - vs.ravel() / (h - 1)], axis=-1)

# ── Mask-aware face build ────────────────────────────────────
mask_bin = (mask_crop > 127).astype(np.uint8)
mask_dil = cv2.dilate(mask_bin, np.ones((3, 3), np.uint8), iterations=1)
mask_dil = cv2.erode(mask_dil, np.ones((3, 3), np.uint8), iterations=2)

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

if MIRROR_MODE:
    z_flat = z.ravel()
    z_mean = z_flat[mask_bin.ravel() > 0].mean()

    z_back = (z_flat + BACK_THICKNESS - 0.3 * (z_flat - z_mean)).reshape(h, w)
    z_back = np.clip(z_back, z.min(), z.max() + BACK_THICKNESS + 0.5)

    front_verts = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)
    back_verts  = np.stack([X.ravel(), Y.ravel(), z_back.ravel()], axis=-1)
    vertices = np.vstack([front_verts, back_verts])
    uv = np.vstack([uv, uv])

    off = h * w
    back_faces = np.vstack([
        np.stack([a + off, c + off, b + off], axis=1),
        np.stack([a + off, d + off, c + off], axis=1),
    ])

    contours, _ = cv2.findContours(mask_dil, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(c, epsilon=2.0, closed=True) for c in contours]

    edges = []
    for cnt in contours:
        pts = cnt[:, 0, :]
        n = len(pts)
        if n < 3:
            continue
        for k in range(n):
            x1_, y1_ = pts[k]
            x2_, y2_ = pts[(k + 1) % n]
            f1 = y1_ * w + x1_
            f2 = y2_ * w + x2_
            edges.append([f1, f2, f2 + off])
            edges.append([f1, f2 + off, f1 + off])
    edges = np.array(edges, dtype=np.int64)
    faces = np.vstack([front_faces, back_faces, edges])
else:
    vertices = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)
    faces = front_faces.astype(np.int64)

print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

# ── FIX 3: Laplacian mesh smoothing (turns stripes into curves) ─
print(f"\nApplying {SMOOTH_ITERS} Laplacian smoothing iterations...")
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
try:
    trimesh.smoothing.filter_laplacian(mesh, lamb=SMOOTH_ITERS * 0.05)
    print("Laplacian smoothing applied")
except Exception as e:
    print(f"Laplacian smoothing skipped: {e}")

mesh.remove_unreferenced_vertices()
print(f"After cleanup: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

# ── Stage 4: Export GLB ──────────────────────────────────────
print("\n── STAGE 4: Export GLB ──")

texture_img = Image.fromarray(rgb_crop)
visual = trimesh.visual.TextureVisuals(
    uv=uv,
    image=texture_img,
    material=trimesh.visual.texture.SimpleMaterial(image=texture_img),
)
mesh.visual = visual

label_str = "+".join(labels)
suffix = "closed" if MIRROR_MODE else "open"
glb_path = os.path.join(OUTPUT_DIR, f"{label_str}_{base_name}_{suffix}.glb")
mesh.export(glb_path)
print(f"Saved: {glb_path}")
print(f"Size: {os.path.getsize(glb_path)/1024/1024:.2f} MB")

print("\n── Contents of artifacts2/ ──")
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p)/1024:.1f} KB)")

print("\nDONE.")
