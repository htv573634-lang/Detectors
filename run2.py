import os
import glob
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import pipeline
import trimesh
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────
INPUT_DIR  = "inputs2"
OUTPUT_DIR = "artifacts2"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SMPL_ONNX   = "models/smpl.onnx"
ROMP_ONNX   = "models/romp.onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Find image ────────────────────────────────────────────────
EXTS = ("*.jpg","*.jpeg","*.jpge","*.png","*.bmp","*.webp")
files = []
for e in EXTS:
    files += glob.glob(os.path.join(INPUT_DIR, e))
    files += glob.glob(os.path.join(INPUT_DIR, e.upper()))
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")

IMAGE_PATH = files[0]
base_name  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_bgr = cv2.cvtColor(np.array(Image.open(IMAGE_PATH).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── Stage 1: ROMP → pose params (no SMPL file needed) ────────
print("\n── STAGE 1: ROMP body pose ──")

import romp
settings = romp.main.default_settings
settings.mode = "image"
settings.show = False
settings.onnx = True
settings.calc_smpl = False          # ← KEY FIX: skip SMPL parser

romp_model = romp.ROMP(settings)
outputs = romp_model(img_bgr)

if isinstance(outputs, dict):
    outputs = [outputs]
person = max(outputs, key=lambda x: float(np.max(x.get("center_conf", [1]))))

betas        = np.asarray(person["betas"]).reshape(1, -1).astype(np.float32)
body_pose    = np.asarray(person["pose"]).reshape(1, -1, 3).astype(np.float32)
global_orient = np.asarray(person.get("global_orient",
                            np.zeros((1, 1, 3)))).reshape(1, 1, 3).astype(np.float32)

if body_pose.shape[1] == 24:
    global_orient = body_pose[:, :1, :]
    body_pose     = body_pose[:, 1:, :]

print(f"betas: {betas.shape}, body_pose: {body_pose.shape}, orient: {global_orient.shape}")

# ── Stage 2: NoSMPL → solid mesh (no SMPL file needed) ──────
print("\n── STAGE 2: NoSMPL solid mesh ──")

from nosmpl.smpl_onnx import SMPLOnnxRuntime
smpl_onnx = SMPLOnnxRuntime(SMPL_ONNX)

smpl_out = smpl_onnx.forward(body_pose, global_orient, betas)

if isinstance(smpl_out, dict):
    vertices = np.asarray(smpl_out.get("vertices",
                     smpl_out.get("verts")))[0]
    faces    = np.asarray(smpl_out.get("faces"))
elif isinstance(smpl_out, (list, tuple)):
    vertices, faces = np.asarray(smpl_out[0])[0], np.asarray(smpl_out[1])
else:
    vertices, faces = np.asarray(smpl_out)[0], smpl_onnx.faces

print(f"SMPL mesh: {vertices.shape[0]} verts, {faces.shape[0]} faces")
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# ── Stage 3: Depth Anything → surface detail ─────────────────
print("\n── STAGE 3: Depth Anything surface detail ──")

depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device="cpu")
res = depth_pipe(Image.fromarray(img_rgb))
if isinstance(res, list):
    res = res[0]

if "predicted_depth" in res:
    raw = res["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(res["depth"]).astype(np.float32)

lo, hi = np.percentile(raw, 2), np.percentile(raw, 98)
disp = np.clip((raw - lo) / (hi - lo + 1e-8), 0, 1)

H, W = disp.shape
verts = mesh.vertices.copy()
norms = mesh.vertex_normals.copy()

x_norm = (verts[:,0] - verts[:,0].min()) / (verts[:,0].ptp() + 1e-8)
y_norm = (verts[:,1] - verts[:,1].min()) / (verts[:,1].ptp() + 1e-8)
px = np.clip((x_norm * (W-1)).astype(int), 0, W-1)
py = np.clip((y_norm * (H-1)).astype(int), 0, H-1)
depth_at_vertex = disp[py, px]

front = norms[:,2] > 0
verts[front,2] = depth_at_vertex[front] * 0.3 + verts[front,2] * 0.7

mesh.vertices = verts
mesh.compute_vertex_normals()

# ── Stage 4: Export GLB ──────────────────────────────────────
glb_path = os.path.join(OUTPUT_DIR, f"human_{base_name}.glb")
mesh.export(glb_path)
print(f"\nSaved: {glb_path}")
print("DONE.")
