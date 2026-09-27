import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from transformers import pipeline
import trimesh
import open3d as o3d

# ── Config ────────────────────────────────────────────────────
INPUT_DIR  = "inputs2"
OUTPUT_DIR = "artifacts2"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

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

# ── Stage 1: MediaPipe Holistic → detect any body part ──────
print("\n── STAGE 1: Detect body part ──")

mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3) as holistic:
    result = holistic.process(img_rgb)

def landmarks_to_bbox(landmarks, W, H, pad=25):
    xs = [lm.x * W for lm in landmarks.landmark]
    ys = [lm.y * H for lm in landmarks.landmark]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(W, int(max(xs)) + pad)
    y1 = min(H, int(max(ys)) + pad)
    return (x0, y0, x1, y1)

detected = []
if result.face_landmarks:
    detected.append(("face", landmarks_to_bbox(result.face_landmarks, W, H)))
if result.left_hand_landmarks:
    detected.append(("left_hand", landmarks_to_bbox(result.left_hand_landmarks, W, H)))
if result.right_hand_landmarks:
    detected.append(("right_hand", landmarks_to_bbox(result.right_hand_landmarks, W, H)))
if result.pose_landmarks:
    detected.append(("body", landmarks_to_bbox(result.pose_landmarks, W, H, pad=15)))

if detected:
    # Merge all detected regions into one bbox
    x0 = min(b[0] for _, b in detected)
    y0 = min(b[1] for _, b in detected)
    x1 = max(b[2] for _, b in detected)
    y1 = max(b[3] for _, b in detected)
    labels = "+".join(l for l, _ in detected)
    print(f"Detected: {labels}")
    print(f"Merged bbox: ({x0},{y0}) → ({x1},{y1})")
else:
    print("⚠️  No body part detected — reconstructing full image.")
    x0, y0, x1, y1 = 0, 0, W, H
    labels = "full"

# ── Stage 2: Depth Anything → depth map ──────────────────────
print("\n── STAGE 2: Depth Anything ──")

depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
out = depth_pipe(Image.fromarray(img_rgb))
if isinstance(out, list): out = out[0]

if "predicted_depth" in out:
    raw = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(out["depth"]).astype(np.float32)

# Crop to the detected body part (or full image if none)
depth_crop = raw[y0:y1, x0:x1]

lo, hi = np.percentile(depth_crop, 2), np.percentile(depth_crop, 98)
disp = np.clip((depth_crop - lo) / (hi - lo + 1e-8), 0, 1)

# Optional: mask out background using edge-aware threshold
# Keeps only the region where depth varies (the subject stands out)
grad = np.abs(np.gradient(disp)[0]) + np.abs(np.gradient(disp)[1])
mask = grad > np.percentile(grad, 60)
disp = disp * (0.6 + 0.4 * mask)   # attenuate flat regions

# ── Stage 3: Depth → point cloud → mesh ──────────────────────
print("\n── STAGE 3: Reconstruct mesh ──")

z = 1.0 / (disp + 1e-3)
z = (z - z.min()) / (z.max() - z.min() + 1e-8)

h, w = z.shape
cx, cy = w / 2.0, h / 2.0
us, vs = np.meshgrid(np.arange(w), np.arange(h))
X = (us - cx) * z / w
Y = (vs - cy) * z / w

pts = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
)
pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9, scale=1.1
)
mesh_o3d.remove_degenerate_triangles()
mesh_o3d.remove_duplicated_vertices()
mesh_o3d.remove_duplicated_triangles()
mesh_o3d.compute_vertex_normals()

# ── Stage 4: Export GLB ──────────────────────────────────────
print("\n── STAGE 4: Export GLB ──")

glb_path = os.path.join(OUTPUT_DIR, f"{labels}_{base_name}.glb")
tri = trimesh.Trimesh(
    vertices=np.asarray(mesh_o3d.vertices),
    faces=np.asarray(mesh_o3d.triangles),
    process=False,
)
tri.export(glb_path)
print(f"Saved: {glb_path}")
print("DONE.")
