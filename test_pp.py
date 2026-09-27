import os
import glob
import numpy as np
import cv2
from PIL import Image

INPUT_DIR  = "inputs2"
OUTPUT_DIR = "out_pp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Find image ────────────────────────────────────────────────
files = []
for ext in ("jpg","jpeg","jpge","png","bmp","webp","tif","tiff"):
    files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext}"))
    files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext.upper()}"))
files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")
files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
IMAGE_PATH = files[0]
base_name  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)
if img is None:
    img = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")), cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

# ── Load ONNX model ──────────────────────────────────────────
import onnxruntime as ort
onnx_path = "pphumanseg_lite.onnx"
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"{onnx_path} not found — workflow must download it first")

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
print(f"Input: {inp.name}, shape: {inp.shape}")

# Get expected input size
in_shape = inp.shape
target_h = in_shape[2] if isinstance(in_shape[2], int) else 192
target_w = in_shape[3] if isinstance(in_shape[3], int) else 192
print(f"Expected input: {target_w}×{target_h}")

# ── Preprocess ───────────────────────────────────────────────
resized = cv2.resize(img_rgb, (target_w, target_h))
norm = resized.astype(np.float32) / 255.0
norm = (norm - 0.5) / 0.5                       # mean=0.5, std=0.5
chw = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

# ── Inference ────────────────────────────────────────────────
outputs = session.run(None, {inp.name: chw})
print(f"Output shapes: {[o.shape for o in outputs]}")

logits = outputs[0]
# Handle NCHW or NHWC
if logits.ndim == 4 and logits.shape[1] == 2:
    mask_small = np.argmax(logits[0], axis=0).astype(np.uint8)
elif logits.ndim == 4 and logits.shape[-1] == 2:
    mask_small = np.argmax(logits[0], axis=-1).astype(np.uint8)
else:
    # Fallback: threshold first channel
    mask_small = (logits.reshape(target_h, target_w) > 0).astype(np.uint8)

mask_small = (mask_small == 1).astype(np.uint8) * 255
mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)

# ── Save outputs ─────────────────────────────────────────────
mask_path = os.path.join(OUTPUT_DIR, f"pp_mask_{base_name}.png")
cv2.imwrite(mask_path, mask)

overlay = img_rgb.copy()
overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)
overlay_path = os.path.join(OUTPUT_DIR, f"pp_overlay_{base_name}.png")
cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"Mask saved: {mask_path}")
print(f"Overlay saved: {overlay_path}")
print(f"Mask coverage: {100*mask.mean()/255:.1f}%")
print("DONE.")
