import os
import glob
import sys
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import trimesh
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = "inputs2"
HMR_DIR = "out_hmr2"
OUTPUT_DIR = "out_fusion"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

# Bidirectional displacement (can push in OR out)
DETAIL_STRENGTH = 1.0
BLUR_SIGMA = 45
SMOOTH_ITERS = 6
DETAIL_PCT_LOW = 5
DETAIL_PCT_HIGH = 95
MAX_DISPLACEMENT = 0.04
MAX_GRADIENT = 0.008

# Reliefs (very small)
FACE_RELIEF = 0.008
HAND_RELIEF = 0.003
HAIR_RELIEF = 0.012
SILHOUETTE_PUFF = 0.0

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
vertices = data["vertices"].copy()
faces = data["faces"]
bbox = data["detection_box"]
img_h = int(data["image_h"])
img_w = int(data["image_w"])
print("[INFO] Mesh: " + str(len(vertices)) + " verts, " + str(len(faces)) + " faces")

original_verts = vertices.copy()

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
# STAGE 2: DEPTH DETAIL (WITH EDGE SUPPRESSION)
# =====================================================
print("[INFO] Stage 2: Extracting depth detail...")
d_blur = cv2.GaussianBlur(d, (0, 0), BLUR_SIGMA)
detail_raw = d - d_blur

lo = np.percentile(detail_raw, DETAIL_PCT_LOW)
hi = np.percentile(detail_raw, DETAIL_PCT_HIGH)
detail = np.clip(detail_raw, lo, hi)

std = detail.std() + 1e-8
detail = detail / (std * 2.0)
detail = np.clip(detail, -1.0, 1.0)
detail = cv2.GaussianBlur(detail, (0, 0), 3.0)

# Suppress edges to prevent boundary artifacts
detail_edges = cv2.Canny((detail * 255).astype(np.uint8), 50, 150)
detail_edges = cv2.dilate(detail_edges, np.ones((5, 5), np.uint8), iterations=1)
detail[detail_edges > 0] = 0.0
print("[INFO] Detail edges suppressed: " + str(int((detail_edges > 0).sum())) + " pixels")

print("[INFO] Detail stats: min=" + str(round(float(detail.min()),3))
      + " max=" + str(round(float(detail.max()),3))
      + " std=" + str(round(float(detail.std()),3)))

detail_vis = ((detail + 1.0) * 127.5).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_detail_" + base_name + ".png"), detail_vis)
print("[OK] Detail map saved")

# =====================================================
# STAGE 3: DSINE NORMALS
# =====================================================
dsine_normals = None
if USE_DSINE:
    print("[INFO] Stage 3: DSINE surface normals...")
    dsine_path = "DSINE/projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
    if not os.path.exists(dsine_path):
        print("[WARN] DSINE ckpt missing at " + dsine_path)
    else:
        try:
            import torch
            import torch.nn.functional as F
            dsine_root = os.path.abspath("DSINE")
            if dsine_root not in sys.path:
                sys.path.insert(0, dsine_root)

            DSINE = None
            for import_path in ("dsine.models.DSINE", "models.DSINE"):
                try:
                    mod = __import__(import_path, fromlist=["DSINE"])
                    DSINE = getattr(mod, "DSINE")
                    print("[OK] DSINE imported from " + import_path)
                    break
                except Exception as ie:
                    print("[DEBUG] Failed " + import_path + ": " + str(ie))

            if DSINE is not None:
                model = DSINE()
                ckpt = torch.load(dsine_path, map_location="cpu")
                state = ckpt.get("model", ckpt)
                model.load_state_dict(state, strict=False)
                model.eval()

                img_t = torch.from_numpy(img_rgb).float() / 255.0
                img_t = img_t.permute(2, 0, 1).unsqueeze(0)
                img_t = F.interpolate(img_t, size=(480, 640),
                                      mode="bilinear", align_corners=False)

                with torch.no_grad():
                    dsine_out = model(img_t)
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
# STAGE 4: FACE MESH
# =====================================================
face_landmarks = None
if USE_FACE:
    print("[INFO] Stage 4: MediaPipe Face Mesh...")
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                   refine_landmarks=False,
                                   min_detection_confidence=0.3) as fm:
            fm_result = fm.process(img_rgb)
        if fm_result.multi_face_landmarks:
            face_landmarks = np.array([
                [lm.x * img_w, lm.y * img_h]
                for lm in fm_result.multi_face_landmarks[0].landmark
            ])
            print("[OK] Face Mesh: " + str(len(face_landmarks)) + " landmarks")
        else:
            print("[WARN] No face detected")
    except Exception as e:
        print("[WARN] Face failed: " + str(e))

# =====================================================
# STAGE 5: HANDS
# =====================================================
hand_landmarks = None
if USE_HANDS:
    print("[INFO] Stage 5: MediaPipe Hands...")
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                            min_detection_confidence=0.3) as hands:
            hand_result = hands.process(img_rgb)
        if hand_result.multi_hand_landmarks:
            hand_landmarks = []
            for h in hand_result.multi_hand_landmarks:
                hand_landmarks.append(np.array([
                    [lm.x * img_w, lm.y * img_h] for lm in h.landmark
                ]))
            print("[OK] Detected " + str(len(hand_landmarks)) + " hand(s)")
        else:
            print("[WARN] No hands detected")
    except Exception as e:
        print("[WARN] Hands failed: " + str(e))

# =====================================================
# STAGE 6: HAIR
# =====================================================
hair_mask = None
if USE_HAIR:
    print("[INFO] Stage 6: Hair segmentation...")
    try:
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        hair_model_path = "hair_segmentation.tflite"
        if not os.path.exists(hair_model_path):
            url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
            try:
                urllib.request.urlretrieve(url, hair_model_path)
            except Exception as dl_err:
                print("[WARN] Hair download failed: " + str(dl_err))

        if os.path.exists(hair_model_path):
            base_options = mp_python.BaseOptions(model_asset_path=hair_model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options, output_category_mask=True
            )
            with vision.ImageSegmenter.create_from_options(options) as segmenter:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                seg_result = segmenter.segment(mp_image)
                hair_mask = seg_result.category_mask.numpy_view().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(OUTPUT_DIR, "fusion_hair_" + base_name + ".png"),
                            hair_mask)
                print("[OK] Hair mask computed")
    except Exception as e:
        print("[WARN] Hair failed: " + str(e))

# =====================================================
# STAGE 7: BUILD FUSED MESH (BIDIRECTIONAL)
# =====================================================
print("[INFO] Stage 7: Building fused mesh...")
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.merge_vertices()
mesh.remove_unreferenced_vertices()
_ = mesh.vertex_normals

verts = np.asarray(mesh.vertices).copy()
vnormals = np.asarray(mesh.vertex_normals).copy()

# Project vertices to image
vx, vy = verts[:, 0], verts[:, 1]
x_min, x_max = vx.min(), vx.max()
y_min, y_max = vy.min(), vy.max()
nx = (vx - x_min) / (x_max - x_min + 1e-8)
ny = 1.0 - (vy - y_min) / (y_max - y_min + 1e-8)
bx1, by1, bx2, by2 = bbox
px = np.clip((bx1 + nx * (bx2 - bx1)).astype(int), 0, img_w - 1)
py = np.clip((by1 + ny * (by2 - by1)).astype(int), 0, img_h - 1)

# Choose normals
if dsine_normals is not None:
    use_normals = dsine_normals[py, px]
    nrm = np.linalg.norm(use_normals, axis=1, keepdims=True) + 1e-8
    use_normals = use_normals / nrm
    use_normals[:, 1] *= -1
    print("[INFO] Using DSINE normals")
else:
    use_normals = vnormals
    print("[INFO] Using mesh normals")

# Strict front-facing mask
front_mask = use_normals[:, 2] > 0.4
print("[INFO] Front-facing: " + str(int(front_mask.sum())) + " / " + str(len(verts)))

# =====================================================
# BIDIRECTIONAL DISPLACEMENT
# Compares mesh depth vs image depth per vertex
# Positive = mesh too shallow -> push out
# Negative = mesh too deep  -> push in (slims the body)
# =====================================================
mesh_z = verts[:, 2]
mesh_z_min, mesh_z_max = mesh_z.min(), mesh_z.max()
mesh_z_norm = (mesh_z - mesh_z_min) / (mesh_z_max - mesh_z_min + 1e-8)

# Image depth at each vertex (0 = far, 1 = near)
image_depth = d[py, px]

# Depth difference
depth_diff = image_depth - mesh_z_norm
print("[INFO] Depth diff: min=" + str(round(float(depth_diff.min()), 4))
      + " max=" + str(round(float(depth_diff.max()), 4))
      + " mean=" + str(round(float(depth_diff.mean()), 4)))

# Bidirectional displacement
raw_disp = depth_diff * DETAIL_STRENGTH
raw_disp = np.clip(raw_disp, -MAX_DISPLACEMENT, MAX_DISPLACEMENT)
raw_disp = raw_disp * front_mask.astype(np.float32)

# Tiny silhouette puff (0.0 = disabled)
if SILHOUETTE_PUFF > 0:
    raw_disp = raw_disp + front_mask.astype(np.float32) * SILHOUETTE_PUFF

print("[INFO] Before gradient limit:")
print("  max abs: " + str(round(float(np.abs(raw_disp).max()), 4)))
print("  mean abs: " + str(round(float(np.abs(raw_disp).mean()), 4)))
print("  negative verts (pushed in): " + str(int((raw_disp < -0.001).sum())))
print("  positive verts (pushed out): " + str(int((raw_disp > 0.001).sum())))

# ---- GRADIENT LIMITING ----
print("[INFO] Applying gradient limiting...")
neighbors = defaultdict(set)
for face in mesh.faces:
    a, b, c = int(face[0]), int(face[1]), int(face[2])
    neighbors[a].update([b, c])
    neighbors[b].update([a, c])
    neighbors[c].update([a, b])

for iteration in range(15):
    new_disp = raw_disp.copy()
    changed = 0
    for i in range(len(raw_disp)):
        if not front_mask[i]:
            continue
        nbrs = [n for n in neighbors[i] if front_mask[n]]
        if not nbrs:
            continue
        nbr_avg = float(np.mean([raw_disp[n] for n in nbrs]))
        delta = raw_disp[i] - nbr_avg
        if abs(delta) > MAX_GRADIENT:
            new_disp[i] = nbr_avg + np.sign(delta) * MAX_GRADIENT
            changed += 1
    raw_disp = new_disp
    if changed == 0:
        print("  Converged after " + str(iteration + 1) + " iterations")
        break

print("[INFO] After gradient limit:")
print("  max abs: " + str(round(float(np.abs(raw_disp).max()), 4)))
print("  mean abs: " + str(round(float(np.abs(raw_disp).mean()), 4)))

# ---- ISOLATED SPIKE FILTER ----
spike_threshold = MAX_GRADIENT * 3
spikes_removed = 0
for i in range(len(raw_disp)):
    if not front_mask[i]:
        continue
    nbrs = [n for n in neighbors[i] if front_mask[n]]
    if len(nbrs) < 3:
        raw_disp[i] = 0.0
        spikes_removed += 1
        continue
    nbr_avg = float(np.mean([raw_disp[n] for n in nbrs]))
    if abs(raw_disp[i] - nbr_avg) > spike_threshold:
        raw_disp[i] = nbr_avg
        spikes_removed += 1
print("[INFO] Spike filter: " + str(spikes_removed) + " vertices corrected")

displacement = use_normals * raw_disp[:, None]

# =====================================================
# WIDTH PRESERVATION
# Compare mesh projected width to image bbox width
# =====================================================
print("[INFO] Checking mesh vs image width...")
mask_proj = np.zeros((img_h, img_w), dtype=np.uint8)
mask_proj[py, px] = 255
mask_ys, mask_xs = np.where(mask_proj > 0)
if len(mask_xs) > 0:
    mesh_width = float(mask_xs.max() - mask_xs.min())
    bbox_width = float(bx2 - bx1)
    width_ratio = mesh_width / (bbox_width + 1e-8)
    print("[INFO] Mesh width: " + str(round(mesh_width, 1))
          + " | bbox width: " + str(round(bbox_width, 1))
          + " | ratio: " + str(round(width_ratio, 3)))

    if width_ratio > 1.05:
        scale_factor = 1.0 / width_ratio
        print("[INFO] Mesh wider than image -- scaling X/Y by " + str(round(scale_factor, 3)))
        center_x = (verts[:, 0].min() + verts[:, 0].max()) / 2
        center_y = (verts[:, 1].min() + verts[:, 1].max()) / 2
        verts[:, 0] = center_x + (verts[:, 0] - center_x) * scale_factor
        verts[:, 1] = center_y + (verts[:, 1] - center_y) * scale_factor
    else:
        print("[INFO] Width OK -- no scaling")

# Face relief
if face_landmarks is not None:
    fl_nx = np.clip((face_landmarks[:, 0] - bx1) / (bx2 - bx1), 0, 1)
    fl_ny = np.clip((face_landmarks[:, 1] - by1) / (by2 - by1), 0, 1)
    fmx = x_min + fl_nx * (x_max - x_min)
    fmy = y_max - fl_ny * (y_max - y_min)
    face_count = 0
    for i in range(0, len(fmx), 8):
        dx = verts[:, 0] - fmx[i]
        dy = verts[:, 1] - fmy[i]
        dist = np.sqrt(dx*dx + dy*dy)
        weight = np.exp(-(dist**2) / (2 * 0.025**2))
        mask = weight > 0.3
        if mask.any():
            displacement[mask] += use_normals[mask] * (FACE_RELIEF * weight[mask, None])
            face_count += int(mask.sum())
    print("[INFO] Face relief on " + str(face_count) + " vertex-hits")

# Hand relief
if hand_landmarks is not None:
    hand_count = 0
    for hand in hand_landmarks:
        h_nx = np.clip((hand[:, 0] - bx1) / (bx2 - bx1), 0, 1)
        h_ny = np.clip((hand[:, 1] - by1) / (by2 - by1), 0, 1)
        hmx = x_min + h_nx * (x_max - x_min)
        hmy = y_max - h_ny * (y_max - y_min)
        for i in range(0, len(hmx), 5):
            dx = verts[:, 0] - hmx[i]
            dy = verts[:, 1] - hmy[i]
            dist = np.sqrt(dx*dx + dy*dy)
            weight = np.exp(-(dist**2) / (2 * 0.02**2))
            mask = weight > 0.3
            if mask.any():
                displacement[mask] += use_normals[mask] * (HAND_RELIEF * weight[mask, None])
                hand_count += int(mask.sum())
    print("[INFO] Hand relief on " + str(hand_count) + " vertex-hits")

# Hair relief
if hair_mask is not None:
    hair_val = hair_mask[py, px] / 255.0
    hair_disp = hair_val * HAIR_RELIEF
    displacement += use_normals * hair_disp[:, None]
    n_hair = int((hair_val > 0.5).sum())
    print("[INFO] Hair relief on " + str(n_hair) + " verts")

new_verts = verts + displacement
total_change = float(np.abs(new_verts - verts).max())
print("[INFO] Total max vertex change: " + str(round(total_change, 4)))

mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)

# =====================================================
# STAGE 8: REPAIR + SMOOTH + EXPORT
# =====================================================
print("[INFO] Stage 8: Repair...")
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

final_change = float(np.abs(np.asarray(mesh.vertices) - original_verts).max())
print("[INFO] Final max change vs original: " + str(round(final_change, 4)))

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
