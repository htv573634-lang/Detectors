import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from transformers import pipeline
import trimesh

# ── Config ────────────────────────────────────────────────────
INPUT_DIR   = "inputs2"
OUTPUT_DIR  = "artifacts2"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
Z_AMPLIFY   = 8.0     # how much to boost depth (3 = subtle, 8 = strong, 12 = extreme)

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

# ── Stage 1: Detect body part + build mask ───────────────────
print("\n── STAGE 1: Detect body part ──")

mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3) as holistic:
    result = holistic.process(img_rgb)

# Build a mask: white where the subject is, black elsewhere
mask = np.zeros((H, W), dtype=np.uint8)
labels = []

def draw_landmarks_mask(landmarks, mask, W, H, thickness=25):
    pts = np.array([[int(lm.x * W), int(lm.y * H)] for lm in landmarks.landmark])
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        for p in pts:
            cv2.circle(mask, tuple(p), thickness, 255, -1)

if result.face_landmarks:
    draw_landmarks_mask(result.face_landmarks, mask, W, H)
    labels.append("face")
if result.left_hand_landmarks:
    draw_landmarks_mask(result.left_hand_landmarks, mask, W, H, thickness=30)
    labels.append("left_hand")
if result.right_hand_landmarks:
    draw_landmarks_mask(result.right_hand_landmarks, mask, W, H, thickness=30)
    labels.append("right_hand")
if result.pose_landmarks:
    draw_landmarks_mask(result.pose_landmarks, mask, W, H, thickness=25)
    labels.append("body")

if not labels:
    print("⚠️  No body part detected — using full image.")
    mask[:] = 255
    labels = ["full"]
else:
    print(f"Detected: {'+'.join(labels)}")

# Bounding box of the mask
ys, xs = np.where(mask > 0)
x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
print(f"Subject bbox: ({x0},{y0}) → ({x1},{y1})")

# ── Stage 2: Depth Anything ──────────────────────────────────
print("\n── STAGE 2: Depth Anything ──")

depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
out = depth_pipe(Image.fromarray(img_rgb))
if isinstance(out, list): out = out[0]

if "predicted_depth" in out:
    raw = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(out["depth"]).astype(np.float32)

# Crop to subject bbox
raw_crop  = raw[y0:y1, x0:x1]
mask_crop = mask[y0:y1, x0:x1]

# Normalize depth using only the subject pixels
subject_vals = raw_crop[mask_crop > 0]
lo, hi = np.percentile(subject_vals, 2), np.percentile(subject_vals, 98)
disp = np.clip((raw_crop - lo) / (hi - lo + 1e-8), 0, 1)

# Outside the subject, set neutral depth so it doesn't create a flat plane
disp[mask_crop == 0] = 0.5

# ── Stage 3: Build 2.5D mesh by direct triangulation ─────────
print("\n── STAGE 3: Reconstruct 2.5D mesh ──")

h, w = disp.shape
# Convert disparity (1=near) to z with amplified range
z = (1.0 - disp) * Z_AMPLIFY

# Back-project to a regular grid
us, vs = np.meshgrid(np.arange(w), np.arange(h))
X = (us - w/2) / max(w, h)
Y = (vs - h/2) / max(w, h)

# Flatten and keep only valid mask pixels
valid = mask_crop.flatten() > 0
vertices_all = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)

index_map = -np.ones(h * w, dtype=np.int64)
index_map[valid] = np.arange(valid.sum())
vertices = vertices_all[valid]

# Build faces (two triangles per grid cell where all 4 corners are valid)
faces = []
idx = np.arange(h * w).reshape(h, w)
for i in range(h - 1):
    for j in range(w - 1):
        quad = [idx[i, j], idx[i, j+1], idx[i+1, j+1], idx[i+1, j]]
        if all(index_map[q] >= 0 for q in quad):
            a, b, c, d = [index_map[q] for q in quad]
            faces.append([a, b, c])
            faces.append([a, c, d])

faces = np.array(faces, dtype=np.int64)
print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

# ── Stage 4: Export GLB ──────────────────────────────────────
print("\n── STAGE 4: Export GLB ──")

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
label_str = "+".join(labels)
glb_path = os.path.join(OUTPUT_DIR, f"{label_str}_{base_name}.glb")
mesh.export(glb_path)
print(f"Saved: {glb_path}")
print("DONE.")
