import os, sys, glob, subprocess
import numpy as np
import cv2, torch
from PIL import Image
from transformers import pipeline
import trimesh

INPUT_DIR = "inputs2"
OUTPUT_DIR = "artifacts2"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Find input image ─────────────────────────────────────────
EXTS = ("*.jpg","*.jpeg","*.jpge","*.png","*.bmp","*.webp")
files = []
for e in EXTS:
    files += glob.glob(os.path.join(INPUT_DIR, e)) + glob.glob(os.path.join(INPUT_DIR, e.upper()))
if not files:
    raise FileNotFoundError(f"No image in {INPUT_DIR}/")

IMAGE_PATH = files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

# ── Stage 1: XFormer → solid SMPL mesh ───────────────────────
print("\n── STAGE 1: XFormer body mesh ──")

# Clone XFormer if not already present
if not os.path.isdir("XFormer"):
    subprocess.run(["git","clone","--depth","1",
                    "https://github.com/HuaweiWei/XFormer.git"], check=True)

sys.path.insert(0, "XFormer")

# Import XFormer inference (adjust to the actual API if needed)
try:
    from xformer.inference import XFormerInference
    xformer = XFormerInference()
    smpl_params = xformer.predict(IMAGE_PATH)   # dict with betas, pose, etc.
    print("XFormer inference successful.")
except Exception as e:
    print(f"XFormer not available or failed: {e}")
    print("Falling back to SMPL template — mesh will be plain but solid.")
    smpl_params = None

# Build a solid SMPL mesh from parameters (or fallback template)
# NOTE: you must have SMPL_NEUTRAL.pkl in ./data/
import smplx
body_model = smplx.create("./data", model_type="smpl",
                          gender="neutral", use_pca=False)

if smpl_params is not None:
    betas = torch.tensor(smpl_params["betas"]).float().unsqueeze(0)
    body_pose = torch.tensor(smpl_params["body_pose"]).float().unsqueeze(0)
    global_orient = torch.tensor(smpl_params["global_orient"]).float().unsqueeze(0)
    out = body_model(betas=betas, body_pose=body_pose,
                     global_orient=global_orient)
    vertices = out.vertices.detach().cpu().numpy()[0]
else:
    # neutral template (no pose)
    out = body_model()
    vertices = out.vertices.detach().cpu().numpy()[0]

faces = body_model.faces
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
print(f"SMPL mesh: {len(vertices)} vertices, {len(faces)} faces")

# ── Stage 2: Depth Anything → surface detail ─────────────────
print("\n── STAGE 2: Depth Anything surface detail ──")

img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL)
res = depth_pipe(Image.fromarray(img_rgb))
if isinstance(res, list):
    res = res[0]

if "predicted_depth" in res:
    raw = res["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
else:
    raw = np.array(res["depth"]).astype(np.float32)

# percentile stretch → 0-1 (1 = close)
lo, hi = np.percentile(raw, 2), np.percentile(raw, 98)
disp = np.clip((raw - lo) / (hi - lo + 1e-8), 0, 1)

# ── Stage 3: Project depth onto SMPL front surface ───────────
print("\n── STAGE 3: Fusing depth onto body mesh ──")

# Simple orthographic projection: for each vertex, look up depth
# based on its (x, y) position, and blend it into the front-facing
# vertices to add real surface relief.
H, W = disp.shape
verts = mesh.vertices.copy()
norms = mesh.vertex_normals.copy()

# Normalize vertex XY to image coords
x_norm = (verts[:,0] - verts[:,0].min()) / (verts[:,0].ptp() + 1e-8)
y_norm = (verts[:,1] - verts[:,1].min()) / (verts[:,1].ptp() + 1e-8)
px = np.clip((x_norm * (W-1)).astype(int), 0, W-1)
py = np.clip((y_norm * (H-1)).astype(int), 0, H-1)

depth_at_vertex = disp[py, px]          # 1 = close, 0 = far

# Only modify vertices facing the camera (+Z normal)
front = norms[:,2] > 0
# Push front vertices outward/inward based on depth
verts[front,2] = depth_at_vertex[front] * 0.3 + verts[front,2] * 0.7

mesh.vertices = verts
mesh.compute_vertex_normals()

# ── Stage 4: Export GLB ──────────────────────────────────────
glb_path = os.path.join(OUTPUT_DIR, f"human_{base_name}.glb")
mesh.export(glb_path)
print(f"\nSaved: {glb_path}")
print("DONE.")
