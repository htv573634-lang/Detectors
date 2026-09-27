import os
import glob
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from unittest.mock import MagicMock

# ── PyTorch 2.6+ compatibility patch ─────────────────────────
# torch>=2.6 flipped weights_only default to True, which rejects
# the Lightning checkpoint's pickled omegaconf objects.
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load
print("✓ torch.load patched for PyTorch 2.6+ compatibility")

# ── Stub pyrender + OpenGL BEFORE importing 4D-Humans ────────
# (we don't render anything — we only export meshes)
sys.modules["pyrender"] = MagicMock()
sys.modules["pyrender.light"] = MagicMock()
sys.modules["pyrender.material"] = MagicMock()
sys.modules["pyrender.mesh"] = MagicMock()
sys.modules["pyrender.node"] = MagicMock()
sys.modules["pyrender.scene"] = MagicMock()
sys.modules["pyrender.viewer"] = MagicMock()
sys.modules["OpenGL"] = MagicMock()
sys.modules["OpenGL.GL"] = MagicMock()
print("✓ pyrender/OpenGL stubbed")

# ── Config ────────────────────────────────────────────────────
INPUT_DIR  = "inputs2"
OUTPUT_DIR = "out_hmr2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Find image ────────────────────────────────────────────────
IMAGE_EXTS = (
    "jpg", "jpeg", "jpge", "jpe", "jfif", "jif", "jfi",
    "png", "bmp", "webp", "tif", "tiff",
)
files = []
for ext in IMAGE_EXTS:
    for pattern in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
        files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))

files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

img_cv2 = cv2.imread(IMAGE_PATH)
if img_cv2 is None:
    img_cv2 = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")), cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"Image size: {W}×{H}")

# ── Import HMR2.0 ────────────────────────────────────────────
sys.path.insert(0, "4D-Humans")

from hmr2.configs import get_config
from hmr2.models import HMR2
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig

# ── Setup HMR2.0 model ────────────────────────────────────────
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

CHECKPOINT = "4D-Humans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
CONFIG_PATH = "4D-Humans/logs/train/multiruns/hmr2/0/model_config.yaml"

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT}")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config missing: {CONFIG_PATH}")

ckpt_size = os.path.getsize(CHECKPOINT)
print(f"Checkpoint size: {ckpt_size/1024/1024:.1f} MB")
if ckpt_size < 100_000_000:
    raise RuntimeError(f"Checkpoint too small ({ckpt_size} bytes) — download failed")

print("Loading HMR2.0 model...")
model_cfg = get_config(CONFIG_PATH)
model = HMR2.load_from_checkpoint(
    CHECKPOINT, strict=False, cfg=model_cfg
).to(device)
model.eval()
print("✓ HMR2.0 loaded")

# ── Setup ViTDet detector ────────────────────────────────────
print("Loading ViTDet detector...")
detectron2_cfg = LazyConfig.load(
    "4D-Humans/vendor/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py"
)
detectron2_cfg.train.init_checkpoint = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
    "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

detector = DefaultPredictor_Lazy(detectron2_cfg)
print("✓ ViTDet loaded")

# ── Detect humans ────────────────────────────────────────────
print("Detecting humans...")
det_out = detector(img_cv2)
det_instances = det_out["instances"]
valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

if len(boxes) == 0:
    raise RuntimeError("No person detected in the image")

print(f"✓ Detected {len(boxes)} person(s)")
for i, b in enumerate(boxes):
    print(f"  Box {i}: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")

# ── Run HMR2.0 ───────────────────────────────────────────────
print("\nRunning HMR2.0 inference...")
dataset = ViTDetDataset(model_cfg, img_cv2.copy(), boxes)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

import trimesh

for i, batch in enumerate(dataloader):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)

    pred_vertices = out["pred_vertices"][0].cpu().numpy()
    pred_faces = model.smpl.faces

    print(f"Person {i}: {len(pred_vertices)} verts, {len(pred_faces)} faces")

    # Bounding box check
    vmin, vmax = pred_vertices.min(axis=0), pred_vertices.max(axis=0)
    print(f"  Bounds: min={vmin.round(3).tolist()} max={vmax.round(3).tolist()}")

    # Save GLB
    mesh = trimesh.Trimesh(vertices=pred_vertices, faces=pred_faces, process=False)
    suffix = f"_{i}" if len(boxes) > 1 else ""
    glb_path = os.path.join(OUTPUT_DIR, f"hmr2_mesh_{base_name}{suffix}.glb")
    mesh.export(glb_path, file_type='glb')
    print(f"  Saved: {glb_path}")
    print(f"  Size: {os.path.getsize(glb_path)/1024:.1f} KB")

    # Save OBJ fallback
    obj_path = os.path.join(OUTPUT_DIR, f"hmr2_mesh_{base_name}{suffix}.obj")
    mesh.export(obj_path, file_type='obj')
    print(f"  OBJ: {obj_path}")

print("\n── Contents of out_hmr2/ ──")
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p)/1024:.1f} KB)")

print("\nDONE.")
