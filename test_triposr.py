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
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
TARGET_SIZE = 256
CHUNK_SIZE = 8192

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

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

log.info("=" * 60)
log.info("TRIPOSR PIPELINE")
log.info("=" * 60)
log.info("Input: " + IMAGE_PATH)

# =====================================================
# INJECT SHIM BEFORE IMPORTING TRIPOSR
# =====================================================
try:
    import torchmcubes_shim
    sys.modules["torchmcubes"] = torchmcubes_shim
    log.info("[OK] torchmcubes shim installed")
except ImportError as e:
    log.warning("Shim not found: " + str(e))

sys.path.insert(0, "TripoSR")
from tsr.system import TSR
from tsr.utils import resize_foreground

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Device: " + device)

# =====================================================
# HUMAN MASK DETECTION (rembg -> fallback threshold)
# =====================================================
def detect_foreground_mask(img_rgb):
    """
    Returns a uint8 mask (0 or 255) where 255 = foreground (person/object).
    Tries rembg (U2-Net) first, falls back to threshold if unavailable.
    """
    h, w = img_rgb.shape[:2]
    # --- Try rembg ---
    try:
        from rembg import remove, new_session
        log.info("[*] Using rembg for foreground detection...")
        # 'u2net_human_seg' is specifically trained for humans
        # Use 'u2net' for general objects, 'u2net_human_seg' for people
        session = new_session("u2net_human_seg")
        pil_in = Image.fromarray(img_rgb).convert("RGB")
        pil_out = remove(pil_in, session=session, only_mask=True,
                         post_process_mask=True)
        mask = np.array(pil_out.convert("L"), dtype=np.uint8)
        # Remap to binary
        mask = ((mask > 127) * 255).astype(np.uint8)
        coverage = mask.mean() / 255.0 * 100
        log.info("[OK] rembg mask: {:.1f}% coverage".format(coverage))
        # Sanity: if rembg mask covers less than 2% or more than 98%, fall back
        if 2.0 < coverage < 98.0:
            return mask
        log.warning("rembg mask coverage suspicious ({:.1f}%), falling back".format(coverage))
    except Exception as e:
        log.warning("rembg failed: " + str(e))

    # --- Fallback: threshold-based ---
    log.info("[*] Fallback: threshold-based mask")
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # White background assumption
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Clean up small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Keep only the largest connected component
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        largest = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
        mask = ((labels == largest) * 255).astype(np.uint8)
    if mask.mean() < 5:
        log.warning("Fallback mask nearly empty, using full image")
        mask = np.ones_like(mask) * 255
    log.info("[OK] Threshold mask: {:.1f}% coverage".format(mask.mean() / 255.0 * 100))
    return mask

# =====================================================
# PREPROCESS
# =====================================================
log.info("=" * 60)
log.info("STAGE 1: Preprocessing image")
log.info("=" * 60)
t0 = time.time()

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Human mask detection ---
mask = detect_foreground_mask(img_rgb)

# Save mask for debugging
mask_path = os.path.join(MASK_DIR, base_name + "_mask.png")
cv2.imwrite(mask_path, mask)
log.info("Mask saved: " + mask_path)

# Optional: also save a background-removed RGB preview
rgba_preview = np.dstack([img_rgb, mask])
Image.fromarray(rgba_preview).save(
    os.path.join(MASK_DIR, base_name + "_rgba.png"))

# --- Composite onto grey background for TripoSR ---
rgba = np.dstack([img_rgb, mask])
pil_image = Image.fromarray(rgba)
pil_image = resize_foreground(pil_image, 0.85)

img_np = np.array(pil_image).astype(np.float32) / 255.0
img_comp = (img_np[:, :, :3] * img_np[:, :, 3:4]
            + (1 - img_np[:, :, 3:4]) * 0.5)
img_final = Image.fromarray((img_comp * 255.0).astype(np.uint8))

# Save the final composited image TripoSR will see
img_final.save(os.path.join(MASK_DIR, base_name + "_composited.png"))

log.info("Preprocessed image: " + str(pil_image.size))
log.info("Preprocess time: " + str(round(time.time()-t0, 2)) + "s")

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

log.info("Model loaded on " + device)
log.info("Load time: " + str(round(time.time()-t0, 2)) + "s")

# =====================================================
# INFERENCE
# =====================================================
log.info("=" * 60)
log.info("STAGE 3: Running TripoSR inference")
log.info("=" * 60)
t0 = time.time()

with torch.no_grad():
    scene_codes = model([img_final], device=device)

log.info("Inference time: " + str(round(time.time()-t0, 2)) + "s")
log.info("Scene codes shape: " + str(tuple(scene_codes.shape)))

# =====================================================
# MESH EXTRACTION
# =====================================================
log.info("=" * 60)
log.info("STAGE 4: Extracting mesh")
log.info("=" * 60)
t0 = time.time()

meshes = model.extract_mesh(scene_codes, has_vertex_color=True,
                            resolution=TARGET_SIZE)

log.info("Mesh extraction time: " + str(round(time.time()-t0, 2)) + "s")
log.info("Number of meshes: " + str(len(meshes)))

# =====================================================
# EXPORT
# =====================================================
log.info("=" * 60)
log.info("STAGE 5: Exporting")
log.info("=" * 60)
t0 = time.time()

for i, mesh in enumerate(meshes):
    suffix = "_" + str(i) if len(meshes) > 1 else ""

    glb_path = os.path.join(OUTPUT_DIR,
                            "triposr_" + base_name + suffix + ".glb")
    mesh.export(glb_path)
    size_kb = os.path.getsize(glb_path) / 1024
    log.info("GLB: " + glb_path + " (" + str(round(size_kb, 1)) + " KB)")

    obj_path = os.path.join(OUTPUT_DIR,
                            "triposr_" + base_name + suffix + ".obj")
    mesh.export(obj_path)
    size_kb = os.path.getsize(obj_path) / 1024
    log.info("OBJ: " + obj_path + " (" + str(round(size_kb, 1)) + " KB)")

    log.info("Mesh " + str(i) + ": " + str(len(mesh.vertices)) + " verts, "
             + str(len(mesh.faces)) + " faces")

log.info("Export time: " + str(round(time.time()-t0, 2)) + "s")
log.info("=" * 60)
log.info("TRIPOSR DONE")
log.info("=" * 60)
