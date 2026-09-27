import os
import cv2
import sys
import numpy as np
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

def log(msg):
    print(msg, flush=True)

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ==========================================
# CLEANUP PREVIOUS ARTIFACTS
# ==========================================
log("Cleaning up previous artifacts...")
for f in os.listdir("artifacts"):
    file_path = os.path.join("artifacts", f)
    if os.path.isfile(file_path):
        os.remove(file_path)
log("Artifacts folder cleared.")
# ==========================================

log("="*60)
log("TRUE 3D SKELETON BUILDER (Root-Relative & Scaled)")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

# ==========================================
# LOAD MODELS
# ==========================================
log("Loading YOLO11-Large Pose...")
model = YOLO('yolo11l-pose.pt') 

log("Loading Depth Anything V2 (Small)...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Indices for key joints
IDX_L_HIP = 11
IDX_R_HIP = 12
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 6

model_name = "real-3d-skeleton"

for img_name in images:
    img_path = os.path.join("inputs", img_name)
    log(f"\nProcessing: {img_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        log(f"Could not read image: {img_name}")
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # ==========================================
    # STAGE 1: GET DEPTH & 2D POSE
    # ==========================================
    log("\n--- STAGE 1: CAMERA-SPACE 3D ---")
    pil_img = Image.fromarray(img_rgb)
    depth_result = depth_pipe(pil_img)
    depth_map = np.array(depth_result["depth"])
    
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    if d_max - d_min > 0:
        depth_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_norm = depth_map
        
    results = model(img_path, augment=True)
    
    if not (results[0].keypoints and results[0].keypoints.xy is not None):
        log("No human detected.")
        continue
        
    kpts = results[0].keypoints.xy[0]
    confs = results[0].keypoints.conf[0]
    
    # Build initial Camera-Space 3D array [X, Y, Z]
    camera_3d = np.zeros((17, 3))
    for i in range(17):
        px = max(0, min(w - 1, int(kpts[i][0])))
        py = max(0, min(h - 1, int(kpts[i][1])))
        # Note: Y is inverted for 3D space (up is positive)
        camera_3d[i] = [kpts[i][0], -kpts[i][1], depth_norm[py, px]]
        
    # ==========================================
    # STAGE 2: CONVERT TO REAL 3D SKELETON
    # ==========================================
    log("\n--- STAGE 2: BUILDING REAL 3D SKELETON ---")
    
    # 1. ROOT THE SKELETON (Move Hips to 0,0,0)
    hip_center = (camera_3d[IDX_L_HIP] + camera_3d[IDX_R_HIP]) / 2.0
    root_relative_3d = camera_3d - hip_center
    
    # 2. SCALE TO REAL WORLD (Meters)
    # Calculate current 3D distance between shoulders
    shoulder_dist = np.linalg.norm(root_relative_3d[IDX_L_SHOULDER] - root_relative_3d[IDX_R_SHOULDER])
    
    # Average human shoulder width is ~0.40 meters. Scale the whole skeleton to match.
    if shoulder_dist > 0:
        scale_factor = 0.40 / shoulder_dist
        real_world_3d = root_relative_3d * scale_factor
    else:
        real_world_3d = root_relative_3d
        
    log("Real-World 3D Skeleton Coordinates (Meters, Rooted at Hips):")
    log("-"*75)
    log(f"{'JOINT':<15} | {'X (Left/Right)':<15} | {'Y (Up/Down)':<15} | {'Z (Front/Back)':<15}")
    log("-"*75)
    
    for i, name in enumerate(keypoint_names):
        x, y, z = real_world_3d[i]
        # Invert Y back so Up is Positive for easier reading
        log(f"{name:<15} | X:{x:<14.4f} | Y:{-y:<14.4f} | Z:{z:<14.4f}")
    log("-"*75)
    
    # Calculate total height (Nose to average Ankle)
    nose_y = -real_world_3d[0][1]
    ankle_y = (-real_world_3d[15][1] + -real_world_3d[16][1]) / 2.0
    estimated_height = nose_y - ankle_y + 0.1 # Add 0.1m for head top
    log(f"Estimated Real-World Height: {estimated_height:.2f} meters")

    # ==========================================
    # SAVE VISUAL ARTIFACTS
    # ==========================================
    base = os.path.splitext(img_name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Depth Map
    depth_8bit = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
    cv2.imwrite(f"artifacts/{base}_depth_{model_name}_{ts}.jpg", depth_colored)
    
    # Save YOLO Skeleton
    if results[0].keypoints:
        annotated = results[0].plot()
        cv2.imwrite(f"artifacts/{base}_skeleton_{model_name}_{ts}.jpg", annotated)
    
    log(f"\nSaved visual artifacts for {img_name}")

log("\nDONE! Check artifacts folder and run logs for the Real 3D Skeleton.")
