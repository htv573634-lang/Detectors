import os
import glob
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh

# ── Config ────────────────────────────────────────────────────
INPUT_DIR   = "inputs2"
OUTPUT_DIR  = "artifacts2"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"
Z_AMPLIFY   = 5.0
MAX_SIDE    = 800
MIRROR_MODE = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Find image ────────────────────────────────────────────────
EXTS = ("*.jpg","*.jpeg","*.jpge","*.png","*.bmp","*.webp","*.tif","*.tiff")
files = []
for e in EXTS:
    files += glob.glob(os.path.join(INPUT_DIR, e))
    files += glob.glob(os.path.join(INPUT_DIR, e.upper()))
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")

IMAGE_PATH = files[0]
base_name  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

# ── Stage 1: Detect body part (Sapiens → MediaPipe fallback) ─
print("\n── STAGE 1: Detect body part ──")

mask = np.zeros((H, W), dtype=np.uint8)
labels = []
sapiens_ok = False

# ---- Try Sapiens ----
try:
    import torch
    from sapiens_inference import (
        SapiensPredictor,
        SapiensConfig,
        SapiensSegmentationType,
    )
    from sapiens_inference.normal import SapiensNormalType
    from sapiens_inference.depth import SapiensDepthType

    sapiens_config = SapiensConfig()
    sapiens_config.device = torch.device("cpu")
    sapiens_config.segmentation_type = SapiensSegmentationType.SEGMENTATION_03B
    sapiens_config.depth_type  = SapiensDepthType.OFF
    sapiens_config.normal_type = SapiensNormalType.OFF

    predictor = SapiensPredictor(sapiens_config)
    seg_result = predictor(img_rgb)

    # Extract the class-ID mask
    if isinstance(seg_result, dict):
        seg_mask_ids = seg_result.get("segmentation",
                        seg_result.get("mask",
                        seg_result.get("seg_map")))
    else:
        seg_mask_ids = seg_result

    seg_mask_ids = np.asarray(seg_mask_ids).astype(np.uint8)
    mask = (seg_mask_ids > 0).astype(np.uint8) * 255

    unique_classes = np.unique(seg_mask_ids)
    if 2 in unique_classes:  labels.append("face")
    if 5 in unique_classes:  labels.append("left_hand")
    if 14 in unique_classes: labels.append("right_hand")
    if 1 in unique_classes or 21 in unique_classes or 22 in unique_classes:
        labels.append("body")
    if not labels:
        labels = ["full"]

    print(f"Sapiens detected classes: {unique_classes}")
    print(f"Sapiens detected parts: {'+'.join(labels)}")
    sapiens_ok = True

except Exception as e:
    print(f"Sapiens unavailable ({type(e).__name__}: {e})")
    print("→ Falling back to MediaPipe Holistic.")

# ---- Fallback: MediaPipe ----
if not sapiens_ok:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
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
        print("⚠️  No body part detected — using full image.")
        mask[:] = 255
        labels = ["full"]
    else:
        print(f"MediaPipe detected: {'+'.join(labels)}")

# ── Smooth mask edges ────────────────────────────────────────
mask = cv2.GaussianBlur(mask, (7, 7), 0)
mask = (mask > 127).astype(np.uint8) * 255

ys, xs = np.where(mask > 0)
x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
print(f"Subject bbox: ({x0},{y0}) → ({x1},{y1})")

# ── Stage 2: Depth Anything ──────────────────────────────────
print("\n── STAGE 2: Depth Anything (Base) ──")

depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
out = depth_pipe(Image.fromarray(img_rgb))
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

subject_vals = raw_crop[mask_crop > 0]
if len(subject_vals) < 10:
    subject_vals = raw_crop.flatten()
lo, hi = np.percentile(subject_vals, 2), np.percentile(subject_vals, 98)
disp = np.clip((raw_crop - lo) / (hi - lo + 1e-8), 0, 1)
print(f"Depth range: {disp.min():.3f} → {disp.max():.3f}, std={disp.std():.3f}")

# ── Stage 3: Downscale then build mesh ───────────────────────
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

idx = np.arange(h * w).reshape(h, w)
a = idx[:-1, :-1].ravel()
b = idx[:-1, 1:].ravel()
c = idx[1:, 1:].ravel()
d = idx[1:, :-1].ravel()

if MIRROR_MODE:
    z_mean = z.mean()
    z_back = 2 * z_mean - z
    z_back = np.clip(z_back, z.min() - 0.5, z.max() + 0.5)

    front_verts = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)
    back_verts  = np.stack([X.ravel(), Y.ravel(), z_back.ravel()], axis=-1)
    vertices = np.vstack([front_verts, back_verts])
    uv = np.vstack([uv, uv])

    front_faces = np.vstack([
        np.stack([a, b, c], axis=1),
        np.stack([a, c, d], axis=1),
    ])

    off = h * w
    back_faces = np.vstack([
        np.stack([a + off, c + off, b + off], axis=1),
        np.stack([a + off, d + off, c + off], axis=1),
    ])

    edges = []
    for j in range(w - 1):
        f0, f1 = idx[0, j], idx[0, j+1]
        edges.append([f0, f0 + off, f1 + off])
        edges.append([f0, f1 + off, f1])
    for j in range(w - 1):
        f0, f1 = idx[h-1, j], idx[h-1, j+1]
        edges.append([f0, f1, f1 + off])
        edges.append([f0, f1 + off, f0 + off])
    for i in range(h - 1):
        f0, f1 = idx[i, 0], idx[i+1, 0]
        edges.append([f0, f1, f1 + off])
        edges.append([f0, f1 + off, f0 + off])
    for i in range(h - 1):
        f0, f1 = idx[i, w-1], idx[i+1, w-1]
        edges.append([f0, f0 + off, f1 + off])
        edges.append([f0, f1 + off, f1])

    edges = np.array(edges, dtype=np.int64)
    faces = np.vstack([front_faces, back_faces, edges])
else:
    vertices = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)
    faces = np.vstack([
        np.stack([a, b, c], axis=1),
        np.stack([a, c, d], axis=1),
    ]).astype(np.int64)

print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

# ── Stage 4: Textured GLB ────────────────────────────────────
print("\n── STAGE 4: Export textured GLB ──")

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

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
print("DONE.")
