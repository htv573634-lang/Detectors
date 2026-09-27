import os
import cv2
import json
import sys
import math
import numpy as np
from datetime import datetime
from PIL import Image
import mediapipe as mp
from transformers import pipeline

def log(msg):
    print(msg, flush=True)

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

log("="*60)
log("STAGE 1 & 2 DETECTOR RUNNING")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]

if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

# ==========================================
# LOAD MODELS
# ==========================================
log("Loading Stage 1: MediaPipe 3D Pose...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

log("Loading Stage 2: Depth Anything V2 (Small)...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model_name = "stage1-2-combo"

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
    # STAGE 1: 3D SKELETON
    # ==========================================
    log("\n--- STAGE 1: 3D ANATOMICAL SKELETON ---")
    results = pose.process(img_rgb)
    
    calculations = {}
    if results.pose_world_landmarks:
        world_landmarks = results.pose_world_landmarks.landmark
        log("3D World Coordinates (Meters):")
        log("-"*60)
        for i, name in enumerate(keypoint_names):
            lm = world_landmarks[i]
            calculations[name] = {
                "X": round(lm.x, 4), "Y": round(lm.y, 4), 
                "Z": round(lm.z, 4), "Vis": round(lm.visibility, 2)
            }
            log(f"{name:<15} | X:{lm.x:<6.4f} | Y:{lm.y:<6.4f} | Z:{lm.z:<6.4f}")
        log("-"*60)
    else:
        log("No human detected for Stage 1.")
        
    # ==========================================
    # STAGE 2: DEPTH CALCULATION
    # ==========================================
    log("\n--- STAGE 2: DEPTH MAP CALCULATION ---")
    log("Calculating 3D volume and distance...")
    
    # FIX: Convert NumPy array to PIL Image for the Hugging Face pipeline
    pil_img = Image.fromarray(img_rgb)
    
    depth_result = depth_pipe(pil_img)
    depth_map = np.array(depth_result["depth"])
    
    # Calculate Depth Statistics
    min_d = float(np.min(depth_map))
    max_d = float(np.max(depth_map))
    mean_d = float(np.mean(depth_map))
    
    # Calculate depth at specific key points
    center_d = float(depth_map[h//2, w//2])
    top_left_d = float(depth_map[h//10, w//10])
    
    log("Depth Statistics (Raw Values):")
    log("-"*60)
    log(f"Minimum Depth (Closest object) : {min_d:.4f}")
    log(f"Maximum Depth (Furthest object): {max_d:.4f}")
    log(f"Average Scene Depth             : {mean_d:.4f}")
    log(f"Depth at Image Center           : {center_d:.4f}")
    log(f"Depth at Top-Left Corner        : {top_left_d:.4f}")
    log(f"Depth Range (Max - Min)         : {max_d - min_d:.4f}")
    log("-"*60)
    
    calculations["depth_stats"] = {
        "min": round(min_d, 4), "max": round(max_d, 4), 
        "mean": round(mean_d, 4), "center": round(center_d, 4)
    }
    
    # Colorize the depth map for visual artifact
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    
    # ==========================================
    # SAVE ARTIFACTS
    # ==========================================
    base = os.path.splitext(img_name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Stage 1 Visual (Skeleton)
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        skeleton_img = img.copy()
        mp_drawing.draw_landmarks(skeleton_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(f"artifacts/{base}_skeleton_{model_name}_{ts}.jpg", skeleton_img)
    
    # Save Stage 2 Visual (Depth Map)
    cv2.imwrite(f"artifacts/{base}_depth_{model_name}_{ts}.jpg", depth_colored)
    
    # Save Raw Data
    json_out = f"artifacts/{base}_data_{model_name}_{ts}.json"
    with open(json_out, "w") as f:
        json.dump(calculations, f, indent=4)
    
    log(f"\nSaved artifacts for {img_name}")

log("\nDONE! Check artifacts folder.")
