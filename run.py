import os
import cv2
import json
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

log("="*60)
log("TRUE 3D FUSION: YOLO11 (2D) + DEPTH ANYTHING (3D)")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

# ==========================================
# LOAD MODELS
# ==========================================
log("Loading YOLO11-Small Pose (Best 2D for occlusions)...")
model = YOLO('yolo11s-pose.pt') 

log("Loading Depth Anything V2 (Small - True 3D)...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# YOLO11 COCO Keypoint Names
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model_name = "true-3d-fusion"

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
    # STAGE 1: GET TRUE DEPTH MAP
    # ==========================================
    log("\n--- STAGE 1: CALCULATING TRUE 3D DEPTH ---")
    pil_img = Image.fromarray(img_rgb)
    depth_result = depth_pipe(pil_img)
    depth_map = np.array(depth_result["depth"])
    
    # Normalize depth to 0.0 (closest) to 1.0 (furthest) for easy reading
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    if d_max - d_min > 0:
        depth_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_norm = depth_map
        
    log(f"Depth Range Normalized: 0.0 (Closest) to 1.0 (Furthest)")

    # ==========================================
    # STAGE 2: YOLO 2D + DEPTH FUSION
    # ==========================================
    log("\n--- STAGE 2: FUSING 2D POSE WITH TRUE DEPTH ---")
    results = model(img_path)
    
    calculations = {}
    
    if results[0].keypoints and results[0].keypoints.xy is not None:
        kpts = results[0].keypoints.xy[0]
        confs = results[0].keypoints.conf[0]
        
        log("True 3D Coordinates (Fused):")
        log("-"*70)
        
        for i, name in enumerate(keypoint_names):
            # Get 2D pixel coordinates from YOLO
            x_2d = float(kpts[i][0])
            y_2d = float(kpts[i][1])
            conf = float(confs[i])
            
            # Ensure coordinates are within image bounds
            px = max(0, min(w - 1, int(x_2d)))
            py = max(0, min(h - 1, int(y_2d)))
            
            # PUNCH THROUGH DEPTH MAP to get TRUE Z
            true_z = float(depth_norm[py, px])
            
            # FIX FOR HIDDEN KNEES/ANKLES:
            # If confidence is low (hidden), we interpolate the depth between Hip and Ankle
            if conf < 0.50 and "knee" in name:
                # Find hip and ankle indices
                hip_idx = keypoint_names.index(name.replace("knee", "hip"))
                ankle_idx = keypoint_names.index(name.replace("knee", "ankle"))
                
                hip_x, hip_y = int(kpts[hip_idx][0]), int(kpts[hip_idx][1])
                ankle_x, ankle_y = int(kpts[ankle_idx][0]), int(kpts[ankle_idx][1])
                
                # Get depth at hip and ankle
                hip_z = float(depth_norm[max(0, min(h-1, hip_y)), max(0, min(w-1, hip_x))])
                ankle_z = float(depth_norm[max(0, min(h-1, ankle_y)), max(0, min(w-1, ankle_x))])
                
                # Average them for the hidden knee
                true_z = (hip_z + ankle_z) / 2.0
                status = " [INTERPOLATED]"
            elif conf < 0.50:
                status = " [LOW CONF]"
            else:
                status = " [VISIBLE]"
            
            calculations[name] = {
                "X_2D": round(x_2d, 2),
                "Y_2D": round(y_2d, 2),
                "Z_TrueDepth": round(true_z, 4),
                "Confidence": round(conf, 2),
                "Status": status.strip()
            }
            
            log(f"{name:<15} | 2D:({x_2d:.0f},{y_2d:.0f}) | True Z:{true_z:.4f} | Conf:{conf:.2f}{status}")
        log("-"*70)
    else:
        log("No human detected by YOLO.")

    # ==========================================
    # SAVE ARTIFACTS
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
    
    # Save Data
    json_out = f"artifacts/{base}_data_{model_name}_{ts}.json"
    with open(json_out, "w") as f:
        json.dump(calculations, f, indent=4)
    
    log(f"\nSaved artifacts for {img_name}")

log("\nDONE! Check artifacts folder.")
