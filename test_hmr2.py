import os
import glob
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from unittest.mock import MagicMock

os.environ["HOME"] = os.getcwd()
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), ".cache", "torch")

_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load
print("[OK] torch.load patched for PyTorch 2.6+")

sys.modules["pyrender"] = MagicMock()
sys.modules["pyrender.light"] = MagicMock()
sys.modules["pyrender.material"] = MagicMock()
sys.modules["pyrender.mesh"] = MagicMock()
sys.modules["pyrender.node"] = MagicMock()
sys.modules["pyrender.scene"] = MagicMock()
sys.modules["pyrender.viewer"] = MagicMock()
sys.modules["OpenGL"] = MagicMock()
sys.modules["OpenGL.GL"] = MagicMock()
print("[OK] pyrender/OpenGL stubbed")

INPUT_DIR   = "inputs2"
OUTPUT_DIR  = "out_hmr2"

SMPL_GENDER = os.environ.get("SMPL_GENDER", "female").lower()
if SMPL_GENDER not in ("male", "female"):
    print("[WARN] Invalid SMPL_GENDER, defaulting to female")
    SMPL_GENDER = "female"
print(f"[INFO] SMPL gender: {SMPL_GENDER}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTS = ("jpg", "jpeg", "jpge", "jpe", "jfif", "jif", "jfi",
              "png", "bmp", "webp", "tif", "tiff")
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
print(f"[INFO] Using image: {IMAGE_PATH}")

img_cv2 = cv2.imread(IMAGE_PATH)
if img_cv2 is None:
    img_cv2 = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"[INFO] Image size: {W}x{H}")

CACHE_DIR = os.path.join(os.getcwd(), ".cache", "4DHumans", "data")
SMPL_DIR  = os.path.join(CACHE_DIR, "smpl")
os.makedirs(SMPL_DIR, exist_ok=True)

GENDER_FILE_MAP = {
    "male":   "SMPL_MALE.pkl",
    "female": "SMPL_FEMALE.pkl",
}
target_smpl = os.path.join(SMPL_DIR, GENDER_FILE_MAP[SMPL_GENDER])

import shutil
src_candidates = [
    f"data/smpl/{GENDER_FILE_MAP[SMPL_GENDER]}",
    f"data/smpl/SMPL_{SMPL_GENDER.upper()}.pkl",
    f"data/smpl/basicModel_{SMPL_GENDER[0]}_lbs_10_207_0_v1.0.0.pkl",
]
copied = False
for src in src_candidates:
    if os.path.exists(src):
        shutil.copy(src, target_smpl)
        print(f"[OK] Copied {src} -> {target_smpl}")
        copied = True
        break

if not copied:
    print("[WARN] No SMPL source found. Tried:")
    for s in src_candidates:
        print(f"   {s}")
    raise FileNotFoundError("SMPL file missing")

for fname in ("smpl_mean_params.npz", "SMPL_to_J19.pkl"):
    ok = False
    for src in (f"data/{fname}", f"data/smpl/{fname}"):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(CACHE_DIR, fname))
            print(f"[OK] Copied {src} -> {CACHE_DIR}/{fname}")
            ok = True
            break
    if not ok:
        print(f"[WARN] {fname} not found")

sys.path.insert(0, "4D-Humans")

from hmr2.configs import get_config
from hmr2.models import HMR2
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"[INFO] Using device: {device}")

CHECKPOINT = "4D-Humans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
CONFIG_PATH = "4D-Humans/logs/train/multiruns/hmr2/0/model_config.yaml"

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT}")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config missing: {CONFIG_PATH}")

ckpt_size = os.path.getsize(CHECKPOINT)
print(f"[INFO] Checkpoint size: {ckpt_size/1024/1024:.1f} MB")
if ckpt_size < 100_000_000:
    raise RuntimeError(f"Checkpoint too small ({ckpt_size} bytes)")

print("[INFO] Loading HMR2.0 config...")
model_cfg = get_config(CONFIG_PATH)

# YACS configs are frozen by default — defrost before modifying
model_cfg.defrost()
model_cfg.SMPL.GENDER = SMPL_GENDER
model_cfg.SMPL.MODEL_PATH = SMPL_DIR
model_cfg.SMPL.MEAN_PARAMS = os.path.join(CACHE_DIR, "smpl_mean_params.npz")
print("[OK] Config defrosted and patched")

print(f"[OK] SMPL gender: {model_cfg.SMPL.GENDER}")
print(f"[OK] SMPL model path: {model_cfg.SMPL.MODEL_PATH}")
print(f"[OK] SMPL mean params: {model_cfg.SMPL.MEAN_PARAMS}")

print("[INFO] Loading HMR2.0 model...")
model = HMR2.load_from_checkpoint(CHECKPOINT, strict=False, cfg=model_cfg).to(device)
model.eval()
print("[OK] HMR2.0 loaded")

print("[INFO] Loading ViTDet detector...")
detectron2_cfg = LazyConfig.load(
    "4D-Humans/vendor/detectron2/projects/ViTDet/configs/COCO/"
    "cascade_mask_rcnn_vitdet_h_75ep.py"
)
detectron2_cfg.train.init_checkpoint = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
    "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

detector = DefaultPredictor_Lazy(detectron2_cfg)
print("[OK] ViTDet loaded")

print("[INFO] Detecting humans...")
det_out = detector(img_cv2)
det_instances = det_out["instances"]
valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

if len(boxes) == 0:
    raise RuntimeError("No person detected")

print(f"[OK] Detected {len(boxes)} person(s)")
for i, b in enumerate(boxes):
    print(f"   Box {i}: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")

print("[INFO] Running HMR2.0 inference...")
dataset = ViTDetDataset(model_cfg, img_cv2.copy(), boxes)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

import trimesh

for i, batch in enumerate(dataloader):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)

    pred_vertices = out["pred_vertices"][0].cpu().numpy()
    pred_faces = model.smpl.faces

    print(f"[INFO] Person {i}: {len(pred_vertices)} verts, {len(pred_faces)} faces")

    vmin = pred_vertices.min(axis=0)
    vmax = pred_vertices.max(axis=0)
    print(f"   Bounds min: {vmin.round(3).tolist()}")
    print(f"   Bounds max: {vmax.round(3).tolist()}")

    mesh = trimesh.Trimesh(vertices=pred_vertices, faces=pred_faces, process=False)
    suffix = f"_{i}" if len(boxes) > 1 else ""

    glb_path = os.path.join(
        OUTPUT_DIR, f"hmr2_{SMPL_GENDER}_mesh_{base_name}{suffix}.glb"
    )
    mesh.export(glb_path, file_type="glb")
    print(f"[OK] GLB: {glb_path} ({os.path.getsize(glb_path)/1024:.1f} KB)")

    obj_path = os.path.join(
        OUTPUT_DIR, f"hmr2_{SMPL_GENDER}_mesh_{base_name}{suffix}.obj"
    )
    mesh.export(obj_path, file_type="obj")
    print(f"[OK] OBJ: {obj_path} ({os.path.getsize(obj_path)/1024:.1f} KB)")

print("\n=== Contents of out_hmr2/ ===")
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p)/1024:.1f} KB)")

print("\nDONE.")
