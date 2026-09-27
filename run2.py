import os
import glob
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import pipeline
import trimesh

# ── Sapiens inference package ─────────────────────────────────
from sapiens_inference import (
    SapiensPredictor,
    SapiensConfig,
    SapiensSegmentationType,
)

# ── Config ────────────────────────────────────────────────────
INPUT_DIR   = "inputs2"
OUTPUT_DIR  = "artifacts2"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"
Z_AMPLIFY   = 5.0          # bumped for more visible curves/folds
MAX_SIDE    = 800          # more vertices than 500, still GitHub-safe
MIRROR_MODE = True

# Sapiens segmentation model
SAPIENS_SEG = SapiensSegmentationType.SEGMENTATION_03B

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

# ── Stage 1: Sapiens body-part segmentation ──────────────────
print("\n── STAGE 1: Sapiens body-part segmentation ──")

sapiens_config = SapiensConfig()
sapiens_config.device = torch.device("cpu")
sapiens_config.segmentation_type = SAPIENS_SEG
# Disable depth/normal inside Sapiens — we use Depth Anything instead
sapiens_config.depth_type = "OFF"
sapiens_config.normal_type = "OFF"

predictor = SapiensPredictor(sapiens_config)

# The predictor returns a dict-like result. The segmentation mask
# is stored as a 2D array of class IDs (0-27).
seg_result = predictor(img_rgb)

# Extract the raw class-ID mask
if isinstance(seg_result, dict) and "segmentation" in seg_result:
    seg_mask_ids = seg_result["segmentation"]
elif hasattr(seg_result, "segmentation"):
    seg_mask_ids = seg_result.segmentation
else:
    # Fallback: the package sometimes returns a combined visualisation
    # and stores the mask under a different key
    seg_mask_ids = seg_result.get("mask", seg_result)

seg_mask_ids = np.asarray(seg_mask_ids).astype(np.uint8)

# Sapiens 28-class body-part scheme:
# 0=Background, 1=Apparel, 2=Face_Neck, 3=Hair, 4=Left_Foot,
# 5=Left_Hand, 6=Left_Lower_Arm, 7=Left_Lower_Leg, 8=Left_Shoe,
# 9=Left_Sock, 10=Left_Upper_Arm, 11=Left_Upper_Leg,
# 12=Lower_Clothing, 13=Right_Foot, 14=Right_Hand,
# 15=Right_Lower_Arm, 16=Right_Lower_Leg, 17=Right_Shoe,
# 18=Right_Sock, 19=Right_Upper_Arm, 20=Right_Upper_Leg,
# 21=Torso, 22=Upper_Clothing, 23=Lower_Lip, 24=Upper_Lip,
# 25=Lower_Teeth, 26=Upper_Teeth, 27=Tongue

# Build a tight foreground mask: anything that is NOT background
mask = (seg_mask_ids > 0).astype(np.uint8) * 255

# Detect which parts are present (for the filename label)
unique_classes = np.unique(seg_mask_ids)
labels = []
if 2 in unique_classes: labels.append("face")
if 5 in unique_classes: labels.append("left_hand")
if 14 in unique_classes: labels.append("right_hand")
if 1 in unique_classes or 21 in unique_classes or 22 in unique_classes:
    labels.append("body")

if not labels:
    labels = ["full"]

print(f"Detected classes: {unique_classes}")
print(f"Detected parts: {'+'.join(labels)}")

# Sapiens masks are already tight — smooth only the edges slightly
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
