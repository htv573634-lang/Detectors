import os
import glob
import sys
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh

# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = "inputs2"
HMR_DIR = "out_hmr2"
OUTPUT_DIR = "out_fusion"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

# Fusion parameters (safe values)
DETAIL_STRENGTH = 0.02
BLUR_SIGMA = 40
SMOOTH_ITERS = 3
DETAIL_PCT_LOW = 5
DETAIL_PCT_HIGH = 95
MAX_DISPLACEMENT = 0.02

# Enhanced models (auto-skip if unavailable)
USE_DSINE = True
USE_FACE = True
USE_HANDS = True
USE_HAIR = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# FIND HMR2.0 PARAMS
# =====================================================
params_files = sorted(
    glob.glob(os.path.join(HMR_DIR, "hmr2_params_*.npz")),
    key=os.path.getmtime, reverse=True
)
if not params_files:
    raise FileNotFoundError("No HMR2.0 params in " + HMR_DIR + "/")
params_path = params_files[0]
print("[INFO] Using params: " + params_path)

data = np.load(params_path, allow_pickle=True)
vertices = data["vertices"]
faces = data["faces"]
bbox = data["detection_box"]
img_h = int(data["image_h"])
img_w = int(data["image_w"])
print("[INFO] Mesh: " + str(len(vertices)) + " verts, " + str(len(faces)) + " faces")

# =====================================================
# FIND IMAGE
# =====================================================
IMAGE_EXTS = ("jpg","jpeg","jpge","png","bmp","webp","tif","tiff")
files = []
for ext in IMAGE_EXTS:
    for pat in ("*." + ext, "*." + ext.upper(), "*." + ext.capitalize()):
        files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
if not files:
    raise FileNotFoundError("No image in " + INPUT_DIR + "/")
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print("[INFO] Image: " + IMAGE_PATH)

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")), cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =====================================================
# STAGE 1: DEPTH ANYTHING V2
# =====================================================
print("[INFO] Stage 1: Depth Anything V2...")
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

d = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_depth_" + base_name + ".png"),
            (d * 255).astype(np.uint8))
print("[OK] Depth map saved")

# =====================================================
# STAGE 2: HIGH-FREQUENCY DEPTH DETAIL
# =====================================================
print("[INFO] Stage 2: Extracting depth detail...")
d_blur = cv2.GaussianBlur(d, (0, 0), BLUR_SIGMA)
detail_raw = d - d_blur

lo = np.percentile(detail_raw, DETAIL_PCT_LOW)
hi = np.percentile(detail_raw, DETAIL_PCT_HIGH)
detail = np.clip(detail_raw, lo, hi)
max_abs = max(abs(lo), abs(hi)) + 1e-8
detail = detail / max_abs
detail = cv2.GaussianBlur(detail, (0, 0), 3.0)
print("[INFO] Detail range: " + str(round(float(detail.min()),4)) + " -> " + str(round(float(detail.max()),4)))

# =====================================================
# STAGE 3: DSINE SURFACE NORMALS
# =====================================================
dsine_normals = None
if USE_DSINE:
    print("[INFO] Stage 3: DSINE surface normals...")
    dsine_path = "DSINE/projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
    if not os.path.exists(dsine_path):
        print("[WARN] DSINE checkpoint not found at " + dsine_path)
    else:
        try:
            import torch
            import torch.nn.functional as F

            # DSINE has two loading paths -- try both
            dsine_model = None
            try:
                from dsine.models.DSINE import DSINE
                dsine_model = DSINE()
            except Exception:
                try:
                    sys.path.insert(0, "DSINE")
                    from models.DSINE import DSINE
                    dsine_model = DSINE()
                except Exception as e2:
                    print("[WARN] Could not import DSINE model: " + str(e2))

            if dsine_model is not None:
                ckpt = torch.load(dsine_path, map_location="cpu")
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                dsine_model.load_state_dict(ckpt, strict=False)
                dsine_model.eval()

                # Preprocess for DSINE
                img_t = torch.from_numpy(img_rgb).float() / 255.0
                img_t = img_t.permute(2, 0, 1).unsqueeze(0)
                # DSINE expects dims divisible by 32, use 480x640
                img_t = F.interpolate(img_t, size=(480, 640), mode="bilinear", align_corners=False)

                with torch.no_grad():
                    dsine_out = dsine_model(img_t)

                if isinstance(dsine_out, tuple):
                    dsine_out = dsine_out[0]

                dsine_normals = dsine_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                dsine_normals = cv2.resize(dsine_normals, (img_w, img_h))

                norm_vis = ((dsine_normals + 1.0) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_normals_" + base_name + ".png"),
                            cv2.cvtColor(norm_vis, cv2.COLOR_RGB2BGR))
                print("[OK] DSINE normals computed")
        except Exception as e:
            print("[WARN] DSINE failed: " + str(e))
            dsine_normals = None

# =====================================================
# STAGE 4: MEDIAPIPE FACE MESH (facial landmarks)
# =====================================================
face_landmarks = None
if USE_FACE:
    print("[INFO] Stage 4: MediaPipe Face Mesh...")
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5) as fm:
            fm_result = fm.process(img_rgb)

        if fm_result.multi_face_landmarks:
            face_landmarks = [
                (lm.x, lm.y, lm.z) for lm in fm_result.multi_face_landmarks[0].landmark
            ]
            print("[OK] MediaPipe Face Mesh: " + str(len(face_landmarks)) + " landmarks")
        else:
            print("[WARN] No face detected")
    except Exception as e:
        print("[WARN] Face Mesh failed: " + str(e))

# =====================================================
# STAGE 5: MEDIAPIPE HANDS
# =====================================================
hand_landmarks = None
if USE_HANDS:
    print("[INFO] Stage 5: MediaPipe Hands...")
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                            min_detection_confidence=0.5) as hands:
            hand_result = hands.process(img_rgb)

        if hand_result.multi_hand_landmarks:
            hand_landmarks = [
                [(lm.x, lm.y, lm.z) for lm in h.landmark]
                for h in hand_result.multi_hand_landmarks
            ]
            print("[OK] Detected " + str(len(hand_landmarks)) + " hand(s)")
        else:
            print("[WARN] No hands detected")
    except Exception as e:
        print("[WARN] Hands detection failed: " + str(e))

# =====================================================
# STAGE 6: MEDIAPIPE HAIR SEGMENTATION
# =====================================================
hair_mask = None
if USE_HAIR:
    print("[INFO] Stage 6: MediaPipe Hair segmentation...")
    try:
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        hair_model_path = "hair_segmentation.tflite"
        if not os.path.exists(hair_model_path):
            url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
            print("[INFO] Downloading hair model...")
            try:
                urllib.request.urlretrieve(url, hair_model_path)
            except Exception as dl_err:
                print("[WARN] Hair model download failed: " + str(dl_err))

        if os.path.exists(hair_model_path):
            base_options = mp_python.BaseOptions(model_asset_path=hair_model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True,
            )
            with vision.ImageSegmenter.create_from_options(options) as segmenter:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                seg_result = segmenter.segment(mp_image)
                hair_mask = seg_result.category_mask.numpy_view().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_hair_" + base_name + ".png"), hair_mask)
                print("[OK] Hair mask computed")
    except Exception as e:
        print("[WARN] Hair segmentation failed: " + str(e))

# =====================================================
# STAGE 7: BUILD FUSED MESH
# =====================================================
print("[INFO] Stage 7: Building fused mesh...")
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()
_ = mesh.vertex_normals

verts = np.asarray(mesh.vertices)

# Project vertices to image coordinates
vx, vy = verts[:, 0], verts[:, 1]
x_min, x_max = vx.min(), vx.max()
y_min, y_max = vy.min(), vy.max()

nx = (vx - x_min) / (x_max - x_min + 1e-8)
ny = 1.0 - (vy - y_min) / (y_max - y_min + 1e-8)

bx1, by1, bx2, by2 = bbox
px = np.clip((bx1 + nx * (bx2 - bx1)).astype(int), 0, img_w - 1)
py = np.clip((by1 + ny * (by2 - by1)).astype(int), 0, img_h - 1)

# Choose normals source for displacement
if dsine_normals is not None:
    use_normals = dsine_normals[py, px]
    norms = np.linalg.norm(use_normals, axis=1, keepdims=True) + 1e-8
    use_normals = use_normals / norms
    use_normals[:, 1] *= -1  # flip Y
    print("[INFO] Using DSINE normals for displacement")
else:
    use_normals = np.asarray(mesh.vertex_normals)
    print("[INFO] Using mesh normals for displacement")

# Sample depth detail at each vertex
vertex_detail = detail[py, px]
front_mask = use_normals[:, 2] > 0.3
print("[INFO] Front-facing vertices: " + str(int(front_mask.sum())) + " / " + str(len(verts)))

# Displace
raw_disp = vertex_detail * DETAIL_STRENGTH
raw_disp = np.clip(raw_disp, -MAX_DISPLACEMENT, MAX_DISPLACEMENT)
raw_disp = raw_disp * front_mask.astype(np.float32)
displacement = use_normals * raw_disp[:, None]

# Hair puff
if hair_mask is not None:
    hair_val = hair_mask[py, px] / 255.0
    hair_disp = hair_val * 0.005
    displacement += use_normals * hair_disp[:, None]
    print("[INFO] Hair displacement on " + str(int((hair_val > 0.5).sum())) + " vertices")

new_verts = verts + displacement
mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)

# =====================================================
# STAGE 8: REPAIR + SMOOTH + EXPORT
# =====================================================
print("[INFO] Stage 8: Repairing mesh...")
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()
trimesh.repair.fix_normals(mesh)
trimesh.repair.fix_inversion(mesh)
trimesh.repair.fill_holes(mesh)

try:
    trimesh.smoothing.filter_humphrey(mesh, alpha=0.05, beta=0.3, iterations=SMOOTH_ITERS)
    print("[OK] Smoothing applied")
except Exception as e:
    print("[WARN] Smoothing skipped: " + str(e))

trimesh.repair.fix_normals(mesh)

mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
mesh.faces = np.asarray(mesh.faces, dtype=np.int64)

glb_path = os.path.join(OUTPUT_DIR, "fusion_" + base_name + ".glb")
mesh.export(glb_path, file_type="glb")
print("[OK] GLB: " + glb_path + " (" + str(round(os.path.getsize(glb_path)/1024, 1)) + " KB)")

obj_path = os.path.join(OUTPUT_DIR, "fusion_" + base_name + ".obj")
mesh.export(obj_path, file_type="obj")
print("[OK] OBJ: " + obj_path + " (" + str(round(os.path.getsize(obj_path)/1024, 1)) + " KB)")

print()
print("=== Contents of " + OUTPUT_DIR + "/ ===")
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(p):
        print("  " + f + "  (" + str(round(os.path.getsize(p)/1024, 1)) + " KB)")

print()
print("DONE.")
