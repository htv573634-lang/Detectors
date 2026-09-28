import os
import glob
import sys
import time
import logging
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = "inputs2"
HMR_DIR = "out_hmr2"
OUTPUT_DIR = "out_fusion"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

TARGET_VERTICES = 206700

# Displacement
DETAIL_STRENGTH = 1.0
BLUR_SIGMA = 45
SMOOTH_ITERS = 2
DETAIL_PCT_LOW = 5
DETAIL_PCT_HIGH = 95
MAX_DISPLACEMENT = 0.02
MAX_GRADIENT = 0.002

# Relief magnitudes (halved from v5)
FACE_RELIEF = 0.003
FACE_SIGMA = 0.015
HAND_RELIEF = 0.002
HAND_SIGMA = 0.025
HAIR_RELIEF = 0.006

# Overall displacement clamp
TOTAL_DISPLACEMENT_CAP = 0.02

USE_DSINE = True
USE_FACE = True
USE_HANDS = True
USE_HAIR = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# FIND HMR2.0 PARAMS
# =====================================================
params_files = sorted(glob.glob(os.path.join(HMR_DIR, "hmr2_params_*.npz")),
                      key=os.path.getmtime, reverse=True)
if not params_files:
    raise FileNotFoundError("No HMR2.0 params in " + HMR_DIR + "/")
params_path = params_files[0]
log.info("Params: " + params_path)

data = np.load(params_path, allow_pickle=True)
vertices = data["vertices"].copy()
faces = data["faces"]
bbox = data["detection_box"]
img_h = int(data["image_h"])
img_w = int(data["image_w"])
log.info(f"HMR2.0: {len(vertices)} verts, {len(faces)} faces")

# =====================================================
# HIGH-RES MESH PREPARATION
# =====================================================
log.info("=" * 60)
log.info("HIGH-RES MESH PREPARATION")
log.info("=" * 60)
t0 = time.time()

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()
log.info(f"Base: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

subdiv_count = 0
while len(mesh.vertices) < TARGET_VERTICES and subdiv_count < 4:
    mesh = mesh.subdivide()
    subdiv_count += 1
    log.info(f"  After subdiv {subdiv_count}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

if len(mesh.vertices) > TARGET_VERTICES * 1.10:
    target_faces = int(TARGET_VERTICES * 2)
    log.info(f"Decimating to ~{target_faces} faces")
    try:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        log.info(f"  After decimation: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    except Exception as e:
        log.warning(f"Decimation failed: {e}")

log.info(f"FINAL resolution: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
log.info(f"Prep time: {time.time()-t0:.2f}s")

original_verts = np.asarray(mesh.vertices).copy()

# =====================================================
# FIND IMAGE
# =====================================================
IMAGE_EXTS = ("jpg","jpeg","jpge","png","bmp","webp","tif","tiff")
files = []
for ext in IMAGE_EXTS:
    for pat in ("*." + ext, "*." + ext.upper(), "*." + ext.capitalize()):
        files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
log.info("Image: " + IMAGE_PATH)

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")), cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =====================================================
# STAGE 1: DEPTH
# =====================================================
log.info("=" * 60)
log.info("STAGE 1: Depth Anything V2")
log.info("=" * 60)
t0 = time.time()

depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
out = depth_pipe(Image.fromarray(img_rgb))
if isinstance(out, list): out = out[0]

if "predicted_depth" in out:
    raw = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(out["depth"]).astype(np.float32)

if raw.shape != (img_h, img_w):
    raw = cv2.resize(raw, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)

d = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_depth_" + base_name + ".png"),
            (d * 255).astype(np.uint8))
log.info(f"Depth time: {time.time()-t0:.2f}s")

# =====================================================
# STAGE 2: DETAIL
# =====================================================
log.info("=" * 60)
log.info("STAGE 2: Depth Detail")
log.info("=" * 60)
t0 = time.time()

d_blur = cv2.GaussianBlur(d, (0, 0), BLUR_SIGMA)
detail_raw = d - d_blur
lo = np.percentile(detail_raw, DETAIL_PCT_LOW)
hi = np.percentile(detail_raw, DETAIL_PCT_HIGH)
detail = np.clip(detail_raw, lo, hi)
std = detail.std() + 1e-8
detail = detail / (std * 2.0)
detail = np.clip(detail, -1.0, 1.0)
detail = cv2.GaussianBlur(detail, (0, 0), 3.0)

detail_edges = cv2.Canny((detail * 255).astype(np.uint8), 50, 150)
detail_edges = cv2.dilate(detail_edges, np.ones((5, 5), np.uint8), iterations=1)
detail[detail_edges > 0] = 0.0

log.info(f"Detail: min={detail.min():.3f} max={detail.max():.3f} std={detail.std():.3f}")
cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_detail_" + base_name + ".png"),
            ((detail + 1.0) * 127.5).astype(np.uint8))
log.info(f"Detail time: {time.time()-t0:.2f}s")

# =====================================================
# STAGE 3: DSINE
# =====================================================
dsine_normals = None
if USE_DSINE:
    log.info("=" * 60)
    log.info("STAGE 3: DSINE")
    log.info("=" * 60)
    t0 = time.time()
    dsine_path = "DSINE/projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
    if os.path.exists(dsine_path):
        try:
            import torch
            import torch.nn.functional as F
            sys.path.insert(0, os.path.abspath("DSINE"))
            DSINE = None
            for p in ("dsine.models.DSINE", "models.DSINE"):
                try:
                    m = __import__(p, fromlist=["DSINE"])
                    DSINE = getattr(m, "DSINE")
                    break
                except Exception:
                    continue
            if DSINE:
                model = DSINE()
                ckpt = torch.load(dsine_path, map_location="cpu")
                model.load_state_dict(ckpt.get("model", ckpt), strict=False)
                model.eval()
                img_t = torch.from_numpy(img_rgb).float() / 255.0
                img_t = img_t.permute(2, 0, 1).unsqueeze(0)
                img_t = F.interpolate(img_t, size=(480, 640), mode="bilinear", align_corners=False)
                with torch.no_grad(): o = model(img_t)
                if isinstance(o, tuple): o = o[0]
                dsine_normals = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
                dsine_normals = cv2.resize(dsine_normals, (img_w, img_h))
                log.info("DSINE normals computed")
        except Exception as e:
            log.warning(f"DSINE failed: {e}")
    log.info(f"DSINE time: {time.time()-t0:.2f}s")

# =====================================================
# STAGE 4: FACE
# =====================================================
face_landmarks = None
if USE_FACE:
    log.info("=" * 60)
    log.info("STAGE 4: Face")
    log.info("=" * 60)
    t0 = time.time()
    try:
        import mediapipe as mp
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                              refine_landmarks=False,
                                              min_detection_confidence=0.3) as fm:
            r = fm.process(img_rgb)
        if r.multi_face_landmarks:
            face_landmarks = np.array([[lm.x * img_w, lm.y * img_h]
                                       for lm in r.multi_face_landmarks[0].landmark])
            log.info(f"Face: {len(face_landmarks)} landmarks")
    except Exception as e:
        log.warning(f"Face failed: {e}")
    log.info(f"Face time: {time.time()-t0:.2f}s")

# =====================================================
# STAGE 5: HANDS
# =====================================================
hand_landmarks = None
if USE_HANDS:
    log.info("=" * 60)
    log.info("STAGE 5: Hands")
    log.info("=" * 60)
    t0 = time.time()
    try:
        import mediapipe as mp
        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                       min_detection_confidence=0.3) as hands:
            r = hands.process(img_rgb)
        if r.multi_hand_landmarks:
            hand_landmarks = [np.array([[lm.x * img_w, lm.y * img_h] for lm in h.landmark])
                              for h in r.multi_hand_landmarks]
            log.info(f"Hands: {len(hand_landmarks)}")
    except Exception as e:
        log.warning(f"Hands failed: {e}")
    log.info(f"Hands time: {time.time()-t0:.2f}s")

# =====================================================
# STAGE 6: HAIR
# =====================================================
hair_mask = None
if USE_HAIR:
    log.info("=" * 60)
    log.info("STAGE 6: Hair")
    log.info("=" * 60)
    t0 = time.time()
    try:
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        hair_model_path = "hair_segmentation.tflite"
        if not os.path.exists(hair_model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
                hair_model_path)
        if os.path.exists(hair_model_path):
            base_options = mp_python.BaseOptions(model_asset_path=hair_model_path)
            options = vision.ImageSegmenterOptions(base_options=base_options,
                                                   output_category_mask=True)
            with vision.ImageSegmenter.create_from_options(options) as segmenter:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                seg = segmenter.segment(mp_image)
                hair_mask = seg.category_mask.numpy_view().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_hair_" + base_name + ".png"), hair_mask)
                log.info(f"Hair coverage: {hair_mask.mean()/255*100:.1f}%")
    except Exception as e:
        log.warning(f"Hair failed: {e}")
    log.info(f"Hair time: {time.time()-t0:.2f}s")

# =====================================================
# STAGE 7: FUSION
# =====================================================
log.info("=" * 60)
log.info("STAGE 7: Fusion")
log.info("=" * 60)
t0 = time.time()

verts = np.asarray(mesh.vertices).copy()
_ = mesh.vertex_normals
vnormals = np.asarray(mesh.vertex_normals).copy()

# Project
vx, vy = verts[:, 0], verts[:, 1]
x_min, x_max = vx.min(), vx.max()
y_min, y_max = vy.min(), vy.max()
nx = (vx - x_min) / (x_max - x_min + 1e-8)
ny = 1.0 - (vy - y_min) / (y_max - y_min + 1e-8)
bx1, by1, bx2, by2 = bbox
px = np.clip((bx1 + nx * (bx2 - bx1)).astype(int), 0, img_w - 1)
py = np.clip((by1 + ny * (by2 - by1)).astype(int), 0, img_h - 1)

if dsine_normals is not None:
    use_normals = dsine_normals[py, px]
    nrm = np.linalg.norm(use_normals, axis=1, keepdims=True) + 1e-8
    use_normals = use_normals / nrm
    use_normals[:, 1] *= -1
    log.info("Using DSINE normals")
else:
    use_normals = vnormals
    log.info("Using mesh normals")

front_mask = use_normals[:, 2] > 0.4
log.info(f"Front-facing: {front_mask.sum()} / {len(verts)}")

# Base displacement
mesh_z_norm = (verts[:, 2] - verts[:, 2].min()) / (verts[:, 2].max() - verts[:, 2].min() + 1e-8)
image_depth = d[py, px]
depth_diff = image_depth - mesh_z_norm
raw_disp = depth_diff * DETAIL_STRENGTH
raw_disp = np.clip(raw_disp, -MAX_DISPLACEMENT, MAX_DISPLACEMENT)
raw_disp = raw_disp * front_mask.astype(np.float32)

log.info(f"Depth diff: min={depth_diff.min():.4f} max={depth_diff.max():.4f} mean={depth_diff.mean():.4f}")
log.info(f"Disp: max={np.abs(raw_disp).max():.4f} mean={np.abs(raw_disp).mean():.4f}")
log.info(f"Verts in: {(raw_disp < -0.001).sum()} | out: {(raw_disp > 0.001).sum()}")

# Vectorized gradient limiting
log.info("Building adjacency...")
from scipy.sparse import csr_matrix
faces_arr = np.asarray(mesh.faces)
rows = np.concatenate([faces_arr[:,0], faces_arr[:,1], faces_arr[:,2]])
cols = np.concatenate([faces_arr[:,1], faces_arr[:,2], faces_arr[:,0]])
V = len(verts)
adj = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(V, V))
adj = adj + adj.T
adj.data = np.ones_like(adj.data)
neighbor_count = np.asarray(adj.sum(axis=1)).flatten()
log.info(f"Adjacency built. Avg neighbors: {neighbor_count.mean():.1f}")

log.info("Gradient limiting (vectorized)...")
for iteration in range(20):
    neighbor_avg = adj.dot(raw_disp) / (neighbor_count + 1e-8)
    delta = raw_disp - neighbor_avg
    clamped = np.clip(delta, -MAX_GRADIENT, MAX_GRADIENT)
    new_disp = neighbor_avg + clamped
    new_disp = new_disp * front_mask.astype(np.float32)
    change = float(np.abs(new_disp - raw_disp).max())
    raw_disp = new_disp
    if change < 1e-6:
        log.info(f"Converged after {iteration+1} iterations")
        break

log.info(f"After gradient limit: max={np.abs(raw_disp).max():.4f} mean={np.abs(raw_disp).mean():.4f}")

# Vectorized spike filter
neighbor_avg2 = adj.dot(raw_disp) / (neighbor_count + 1e-8)
delta2 = raw_disp - neighbor_avg2
spike_mask = np.abs(delta2) > (MAX_GRADIENT * 3)
raw_disp[spike_mask] = neighbor_avg2[spike_mask]
log.info(f"Spike filter: {spike_mask.sum()} verts corrected")

displacement = use_normals * raw_disp[:, None]

# FACE RELIEF (single weight map - no stacking)
if face_landmarks is not None:
    face_weight = np.zeros(len(verts), dtype=np.float32)
    fl_nx = np.clip((face_landmarks[:, 0] - bx1) / (bx2 - bx1), 0, 1)
    fl_ny = np.clip((face_landmarks[:, 1] - by1) / (by2 - by1), 0, 1)
    fmx = x_min + fl_nx * (x_max - x_min)
    fmy = y_max - fl_ny * (y_max - y_min)

    for i in range(len(fmx)):
        dx = verts[:, 0] - fmx[i]
        dy = verts[:, 1] - fmy[i]
        dist_sq = dx*dx + dy*dy
        w = np.exp(-dist_sq / (2 * FACE_SIGMA**2))
        face_weight = np.maximum(face_weight, w.astype(np.float32))

    face_mask = face_weight > 0.3
    face_hits = int(face_mask.sum())
    displacement[face_mask] += use_normals[face_mask] * (FACE_RELIEF * face_weight[face_mask, None])
    log.info(f"Face relief hits: {face_hits} (weight max: {face_weight.max():.3f})")

# HAND RELIEF (single weight map - no stacking)
if hand_landmarks is not None:
    hand_weight = np.zeros(len(verts), dtype=np.float32)
    for hand in hand_landmarks:
        h_nx = np.clip((hand[:, 0] - bx1) / (bx2 - bx1), 0, 1)
        h_ny = np.clip((hand[:, 1] - by1) / (by2 - by1), 0, 1)
        hmx = x_min + h_nx * (x_max - x_min)
        hmy = y_max - h_ny * (y_max - y_min)
        for i in range(len(hmx)):
            dx = verts[:, 0] - hmx[i]
            dy = verts[:, 1] - hmy[i]
            dist_sq = dx*dx + dy*dy
            w = np.exp(-dist_sq / (2 * HAND_SIGMA**2))
            hand_weight = np.maximum(hand_weight, w.astype(np.float32))

    hand_mask = hand_weight > 0.3
    hand_hits = int(hand_mask.sum())
    displacement[hand_mask] += use_normals[hand_mask] * (HAND_RELIEF * hand_weight[hand_mask, None])
    log.info(f"Hand relief hits: {hand_hits} (weight max: {hand_weight.max():.3f})")

# HAIR RELIEF
if hair_mask is not None:
    hair_val = hair_mask[py, px].astype(np.float32) / 255.0
    displacement += use_normals * ((hair_val * HAIR_RELIEF)[:, None])
    log.info(f"Hair relief on {(hair_val > 0.5).sum()} verts")

# GLOBAL DISPLACEMENT CAP
disp_magnitude = np.linalg.norm(displacement, axis=1)
over_cap = disp_magnitude > TOTAL_DISPLACEMENT_CAP
if over_cap.any():
    scale = TOTAL_DISPLACEMENT_CAP / (disp_magnitude[over_cap] + 1e-8)
    displacement[over_cap] = displacement[over_cap] * scale[:, None]
    log.info(f"Total displacement cap: {over_cap.sum()} verts scaled down")

log.info(f"Disp magnitude: max={disp_magnitude.max():.4f} mean={disp_magnitude.mean():.4f}")

new_verts = verts + displacement
total_change = float(np.abs(new_verts - verts).max())
log.info(f"Total max change: {total_change:.4f}")

mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)
log.info(f"Fusion time: {time.time()-t0:.2f}s")

# =====================================================
# STAGE 8: REPAIR + EXPORT
# =====================================================
log.info("=" * 60)
log.info("STAGE 8: Repair + Export")
log.info("=" * 60)
t0 = time.time()

# Log mesh change BEFORE repair
change_before_repair = float(np.abs(np.asarray(mesh.vertices) - original_verts).max())
log.info(f"Change BEFORE repair: {change_before_repair:.4f}")

mesh.merge_vertices()
mesh.remove_unreferenced_vertices()

change_after_merge = float(np.abs(np.asarray(mesh.vertices) - original_verts).max())
log.info(f"Change after merge_vertices: {change_after_merge:.4f}")

trimesh.repair.fix_normals(mesh)
change_after_normals = float(np.abs(np.asarray(mesh.vertices) - original_verts).max())
log.info(f"Change after fix_normals: {change_after_normals:.4f}")

try:
    trimesh.repair.fill_holes(mesh)
    change_after_holes = float(np.abs(np.asarray(mesh.vertices) - original_verts).max())
    log.info(f"Change after fill_holes: {change_after_holes:.4f}")
except Exception as e:
    log.warning(f"fill_holes skipped: {e}")

# Use laplacian (NOT humphrey) - humphrey amplifies displacement 9x!
try:
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.3, iterations=SMOOTH_ITERS)
    change_after_smooth = float(np.abs(np.asarray(mesh.vertices) - original_verts).max())
    log.info(f"Smoothing applied. Change after smoothing: {change_after_smooth:.4f}")
except Exception as e:
    log.warning(f"Smoothing skipped: {e}")

trimesh.repair.fix_normals(mesh)

mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
mesh.faces = np.asarray(mesh.faces, dtype=np.int64)

final_change = float(np.abs(np.asarray(mesh.vertices) - original_verts).max())
log.info(f"Final change vs base: {final_change:.4f}")
log.info(f"Final mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

glb_path = os.path.join(OUTPUT_DIR, "fusion_" + base_name + ".glb")
mesh.export(glb_path, file_type="glb")
log.info(f"GLB: {glb_path} ({os.path.getsize(glb_path)/1024/1024:.2f} MB)")

obj_path = os.path.join(OUTPUT_DIR, "fusion_" + base_name + ".obj")
mesh.export(obj_path, file_type="obj")
log.info(f"OBJ: {obj_path} ({os.path.getsize(obj_path)/1024/1024:.2f} MB)")

log.info(f"Export time: {time.time()-t0:.2f}s")
log.info("=" * 60)
log.info("DONE")
log.info("=" * 60)
