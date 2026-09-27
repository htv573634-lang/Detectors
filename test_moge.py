import os
import glob
import numpy as np
import cv2
import torch
from PIL import Image

INPUT_DIR  = "inputs2"
OUTPUT_DIR = "out_moge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Find image ────────────────────────────────────────────────
files = []
for ext in ("jpg","jpeg","jpge","png","bmp","webp","tif","tiff"):
    files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext}"))
    files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext.upper()}"))
files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

# ── Load MoGe model ──────────────────────────────────────────
from moge.model.v2 import MoGeModel

device = torch.device("cpu")
print("Loading MoGe-2 (VitL)...")
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device)
model.eval()

# ── Preprocess ───────────────────────────────────────────────
img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
img_tensor = img_tensor.to(device)

# ── Inference ────────────────────────────────────────────────
print("Running inference...")
with torch.no_grad():
    output = model.infer(img_tensor)

# Output keys: points, depth, mask, normal (per MoGe docs)
depth  = output.get("depth").squeeze().cpu().numpy()
normal = output.get("normal").squeeze().cpu().numpy()  # H,W,3
mask   = output.get("mask").squeeze().cpu().numpy() if "mask" in output else None

print(f"Depth shape: {depth.shape}")
print(f"Normal shape: {normal.shape}")

# ── Save depth ───────────────────────────────────────────────
d = depth.astype(np.float32)
d = (d - d.min()) / (d.max() - d.min() + 1e-8)
d_u8 = (d * 255).astype(np.uint8)
depth_path = os.path.join(OUTPUT_DIR, f"moge_depth_{base_name}.png")
cv2.imwrite(depth_path, d_u8)
print(f"Depth saved: {depth_path}")

# ── Save normals (RGB encoding of XYZ) ───────────────────────
n = normal.astype(np.float32)
n = (n + 1.0) / 2.0                    # [-1,1] -> [0,1]
n_u8 = (np.clip(n, 0, 1) * 255).astype(np.uint8)
normal_path = os.path.join(OUTPUT_DIR, f"moge_normal_{base_name}.png")
cv2.imwrite(normal_path, cv2.cvtColor(n_u8, cv2.COLOR_RGB2BGR))
print(f"Normal saved: {normal_path}")

# ── Save mask if present ─────────────────────────────────────
if mask is not None:
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    mask_path = os.path.join(OUTPUT_DIR, f"moge_mask_{base_name}.png")
    cv2.imwrite(mask_path, m_u8)
    print(f"Mask saved: {mask_path}")

print("DONE.")
