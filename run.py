import os
import cv2
import json
import sys
import numpy as np
from datetime import datetime
from PIL import Image
import mediapipe as mp
from transformers import pipeline

# CRITICAL: Force MediaPipe to use CPU and disable GPU context to prevent crashes
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

def log(msg):
    print(msg, flush=True)

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

log("="*60)
log("LOW-MEMORY 3D POSE & DEPTH (Any Pose Ready)")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

# ==========================================
# LOAD MODELS
# ==========================================
log("Loading MediaPipe 3D (Heavy Model - CPU Only)...")
mp_pose = mp.solutions.pose
# model_complexity=2 loads the heavy 3D prior for complex/sitting poses
pose = mp_pose.Pose(
    static_image_mode=True, 
    model_complexity=2, 
    min_detection_confidence=0.3, # Lowered to catch hidden/occluded joints
    min_tracking_confidence=0.3
)

log("Loading Depth Anything V2 (Small)...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model_name = "lowmem-3d"

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
    # STAGE 1: 3D POSE (Learned Prior)
    # ==========================================
    log("\n--- STAGE 1: 3D POSE (Handling Complex Poses) ---")
    results = pose.process(img_rgb)
    
    calculations = {}
    
    if results.pose_world_landmarks:
        log("3D World Coordinates (Meters) - Relative to Hips:")
        log("-"*70)
        for i, name in enumerate(keypoint_names):
            lm = results.pose_world_landmarks.landmark[i]
            vis = results.pose_landmarks.landmark[i].visibility
            
            calculations[name] = {
                "X_m": round(lm.x, 4), 
                "Y_m": round(lm.y, 4), 
                "Z_m": round(lm.z, 4),
                "Visibility": round(vis, 2)
            }
            
            # Highlight hidden joints
            status = " [HIDDEN/OCCLUDED]" if vis < 0.4 else ""
            log(f"{name:<15} | X:{lm.x:<6.3f} | Y:{lm.y:<6.3f} | Z:{lm.z:<6.3f} | Vis:{vis:.2f}{status}")
        log("-"*70)
    else:
        log("No human detected.")

    # ==========================================
    # STAGE 2: DEPTH MAP
    # ==========================================
    log("\n--- STAGE 2: DEPTH MAP ---")
    pil_img = Image.fromarray(img_rgb)
    depth_result = depth_pipe(pil_img)
    depth_map_raw = np.array(depth_result["depth"])
    
    # Normalize for stats
    d_min = float(np.min(depth_map_raw))
    d_max = float(np.max(depth_map_raw))
    log(f"Depth Range: {d_min:.2f} (Closest) to {d_max:.2f} (Furthest)")
    
    # Colorize
    depth_norm = cv2.normalize(depth_map_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    
    # ==========================================
    # SAVE ARTIFACTS
    # ==========================================
    base = os.path.splitext(img_name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cv2.imwrite(f"artifacts/{base}_depth_{model_name}_{ts}.jpg", depth_colored)
    
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        skeleton_img = img.copy()
        mp_drawing.draw_landmarks(skeleton_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(f"artifacts/{base}_skeleton_{model_name}_{ts}.jpg", skeleton_img)
    
    json_out = f"artifacts/{base}_data_{model_name}_{ts}.json"
    with open(json_out, "w") as f:
        json.dump(calculations, f, indent=4)
    
    log(f"\nSaved artifacts for {img_name}")

log("\nDONE! Check artifacts folder.")
