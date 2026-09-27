import os
import glob
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import open3d as o3d
import trimesh

INPUT_DIR = "inputs2"
OUTPUT_DIR = "artifacts2"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Find any image ----------
EXTS = ("*.jpg", "*.jpeg", "*.jpge", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff")
image_files = []
for ext in EXTS:
    image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))

if not image_files:
    raise FileNotFoundError(f"No image found in {INPUT_DIR}/")

IMAGE_PATH = image_files[0]
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"Using image: {IMAGE_PATH}")

# ---------- Load image ----------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    img_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))
else:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =========================================================
# STAGE 1 : Depth Anything V2
# =========================================================
print(f"\n--- STAGE 1: DEPTH MAP ({DEPTH_MODEL}) ---")

depth_pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL)
depth_result = depth_pipe(Image.fromarray(img_rgb))

if isinstance(depth_result, list):
    depth_result = depth_result[0]

if "predicted_depth" in depth_result:
    raw = depth_result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
elif "depth" in depth_result:
    raw = np.array(depth_result["depth"]).astype(np.float32)
else:
    raise KeyError(f"Unexpected keys: {depth_result.keys()}")

# percentile stretch for stronger relief
lo, hi = np.percentile(raw, 2), np.percentile(raw, 98)
disp = np.clip((raw - lo) / (hi - lo + 1e-8), 0, 1)

# =========================================================
# STAGE 2 : Depth -> mesh -> GLB
# =========================================================
print("\n--- STAGE 2: MESH -> GLB ---")

z = 1.0 / (disp + 1e-3)
z = (z - z.min()) / (z.max() - z.min() + 1e-8)

H, W = z.shape
cx, cy = W / 2.0, H / 2.0
us, vs = np.meshgrid(np.arange(W), np.arange(H))
X = (us - cx) * z / W
Y = (vs - cy) * z / W

pts = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=-1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
)
pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=8, scale=1.1
)
mesh_o3d.remove_degenerate_triangles()
mesh_o3d.remove_duplicated_vertices()
mesh_o3d.remove_duplicated_triangles()
mesh_o3d.compute_vertex_normals()

# ---- Save ONLY the GLB ----
glb_path = os.path.join(OUTPUT_DIR, f"depthanything_{base_name}.glb")
tri = trimesh.Trimesh(
    vertices=np.asarray(mesh_o3d.vertices),
    faces=np.asarray(mesh_o3d.triangles),
    process=False,
)
tri.export(glb_path)
print(f"Saved: {glb_path}")

print("\nDONE. GLB in artifacts2/")
