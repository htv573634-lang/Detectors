import os
import glob
import sys
import time
import logging
import numpy as np
import torch
from PIL import Image
import cv2

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = "inputs"
OUTPUT_DIR = "out_triposr"
TARGET_SIZE = 256  # Mesh extraction resolution
CHUNK_SIZE = 8192  # Lower if OOM on CPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
log.info("=" * 60)
log.info("TRIPOSR PIPELINE")
log.info("=" * 60)
log.info(f"Input: {IMAGE_PATH}")

# =====================================================
# ADD TRIPOSR TO PATH
# =====================================================
sys.path.insert(0, "TripoSR")
from tsr.system import TSR
from tsr.utils import resize_foreground

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Device: {device}")

# =====================================================
# PREPROCESS IMAGE
# =====================================================
log.info("=" * 60)
log.info("STAGE 1: Preprocessing image")
log.info("=" * 60)
t0 = time.time()

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")), cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Create alpha channel (foreground = non-background)
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# Simple threshold: assumes background is white or uniform
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# If mask is mostly empty, use full image
if mask.mean() < 5:
    mask = np.ones_like(mask) * 255
rgba = np.dstack([img_rgb, mask])
pil_image = Image.fromarray(rgba)

# Resize foreground to 85% of frame
pil_image = resize_foreground(pil_image, 0.85)

# Composite on gray background
img_np = np.array(pil_image).astype(np.float32) / 255.0
img_comp = img_np[:, :, :3] * img_np[:, :, 3:4] + (1 - img_np[:, :, 3:4]) * 0.5
img_final = Image.fromarray((img_comp * 255.0).astype(np.uint8))

log.info(f"Preprocessed image: {pil_image.size}")
log.info(f"Preprocess time: {time.time()-t0:.2f}s")

# =====================================================
# LOAD MODEL
# =====================================================
log.info("=" * 60)
log.info("STAGE 2: Loading TripoSR model")
log.info("=" * 60)
t0 = time.time()

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(CHUNK_SIZE)
model.to(device)
model.eval()
log.info(f"Model loaded on {device}")
log.info(f"Load time: {time.time()-t0:.2f}s")

# =====================================================
# INFERENCE
# =====================================================
log.info("=" * 60)
log.info("STAGE 3: Running TripoSR inference")
log.info("=" * 60)
t0 = time.time()

with torch.no_grad():
    scene_codes = model([img_final], device=device)

log.info(f"Inference time: {time.time()-t0:.2f}s")
log.info(f"Scene codes shape: {scene_codes.shape}")

# =====================================================
# MESH EXTRACTION
# =====================================================
log.info("=" * 60)
log.info("STAGE 4: Extracting mesh")
log.info("=" * 60)
t0 = time.time()

meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=TARGET_SIZE)

log.info(f"Mesh extraction time: {time.time()-t0:.2f}s")
log.info(f"Number of meshes: {len(meshes)}")

# =====================================================
# EXPORT
# =====================================================
log.info("=" * 60)
log.info("STAGE 5: Exporting")
log.info("=" * 60)
t0 = time.time()

for i, mesh in enumerate(meshes):
    suffix = f"_{i}" if len(meshes) > 1 else ""
    
    glb_path = os.path.join(OUTPUT_DIR, f"triposr_{base_name}{suffix}.glb")
    mesh.export(glb_path)
    size_kb = os.path.getsize(glb_path) / 1024
    log.info(f"GLB: {glb_path} ({size_kb:.1f} KB)")
    
    obj_path = os.path.join(OUTPUT_DIR, f"triposr_{base_name}{suffix}.obj")
    mesh.export(obj_path)
    size_kb = os.path.getsize(obj_path) / 1024
    log.info(f"OBJ: {obj_path} ({size_kb:.1f} KB)")
    
    # Log mesh stats
    log.info(f"Mesh {i}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

log.info(f"Export time: {time.time()-t0:.2f}s")
log.info("=" * 60)
log.info("TRIPOSR DONE")
log.info("=" * 60)
