import os
import glob
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from unittest.mock import MagicMock

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

INPUT_DIR  = "inputs2"
OUTPUT_DIR = "out_hmr2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Find image ────────────────────────────────────────────────
files = []
for ext in ("jpg", "jpeg", "jpge", "png", "bmp", "webp", "tif", "tiff"):
    files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext}"))
    files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext.upper()}"))
files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")
IMAGE_PATH = files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

img_cv2 = cv2.imread(IMAGE_PATH)
if img_cv2 is None:
    img_cv2 = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")), cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

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

DEFAULT_CHECKPOINT = "4D-Humans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
model_cfg_path = "4D-Humans/logs/train/multiruns/hmr2/0/model_config.yaml"
model_cfg = get_config(model_cfg_path)

model = HMR2.load_from_checkpoint(DEFAULT_CHECKPOINT, strict=False, cfg=model_cfg).to(device)
model.eval()
print("HMR2.0 model loaded.")

# ── Setup ViTDet detector ────────────────────────────────────
detectron2_cfg = LazyConfig.load(
    "4D-Humans/vendor/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py"
)
detectron2_cfg.train.init_checkpoint = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)
print("ViTDet detector loaded.")

# ── Detect humans ────────────────────────────────────────────
det_out = detector(img_cv2)
det_instances = det_out["instances"]
valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

if len(boxes) == 0:
    raise RuntimeError("No person detected in the image")

print(f"Detected {len(boxes)} person(s)")

# ── Run HMR2.0 ───────────────────────────────────────────────
dataset = ViTDetDataset(model_cfg, img_cv2.copy(), boxes)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for batch in dataloader:
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)

    pred_vertices = out["pred_vertices"][0].cpu().numpy()
    pred_faces = model.smpl.faces

    print(f"Mesh: {len(pred_vertices)} verts, {len(pred_faces)} faces")

    # ── Save as GLB ──────────────────────────────────────────
    import trimesh

    mesh = trimesh.Trimesh(vertices=pred_vertices, faces=pred_faces, process=False)
    glb_path = os.path.join(OUTPUT_DIR, f"hmr2_mesh_{base_name}.glb")
    mesh.export(glb_path, file_type='glb')
    print(f"Mesh saved: {glb_path}")
    print(f"Size: {os.path.getsize(glb_path)/1024:.1f} KB")

print("DONE.")
