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

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

def log(msg):
    print(msg, flush=True)

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ==========================================
# DYNAMIC FABRIK INVERSE KINEMATICS SOLVER
# ==========================================
def fabrik_ik_solve(start_pos, end_pos, bone1_len, bone2_len):
    """
    Solves the 3D position of a hidden joint (Knee) between two known joints (Hip and Ankle)
    using the FABRIK algorithm. Handles any pose/angle.
    """
    # Convert to numpy arrays
    p_start = np.array(start_pos, dtype=float)
    p_end = np.array(end_pos, dtype=float)
    
    # Distance between Hip and Ankle
    dist = np.linalg.norm(p_end - p_start)
    total_len = bone1_len + bone2_len
    
    # If the target is out of reach, stretch the leg straight
    if dist >= total_len:
        direction = (p_end - p_start) / dist
        p_mid = p_start + direction * bone1_len
        return p_mid.tolist()
    
    # FABRIK Iteration (Forward and Backward reaching)
    p1, p2 = p_start.copy(), p_end.copy()
    
    for _ in range(10): # 10 iterations is enough for 2 bones
        # Forward reaching
        r = np.linalg.norm(p2 - p1)
        p1 = p1 + (bone2_len / r) * (p2 - p1)
        
        # Backward reaching
        r = np.linalg.norm(p2 - p1)
        p2 = p1 + (bone1_len / r) * (p2 - p1)
        
        # Reset start
        p1 = p_start + (bone1_len / np.linalg.norm(p2 - p_start)) * (p2 - p_start)

    return p1.tolist()

def get_dynamic_bone_lengths(shoulders, hips):
    """
    Calculates leg bone lengths dynamically based on the person's torso size.
    This prevents crashes on different body types (kids vs adults).
    """
    s_l, s_r = np.array(shoulders[0]), np.array(shoulders[1])
    h_l, h_r = np.array(hips[0]), np.array(hips[1])
    
    # Torso length (average distance between shoulders and hips)
    torso_len = (np.linalg.norm(s_l - h_l) + np.linalg.norm(s_r - h_r)) / 2.0
    
    # Anthropometric ratios: Thigh is ~95% of torso, Shin is ~95% of torso
    thigh_len = torso_len * 0.95
    shin_len = torso_len * 0.95
    
    return thigh_len, shin_len

# ==========================================
# MAIN PIPELINE
# ==========================================
log("="*60)
log("DYNAMIC IK 3D POSE & DEPTH (Any Pose / Low Memory)")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

log("Loading MediaPipe 3D (Heavy Model - CPU Only)...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.3)

log("Loading Depth Anything V2 (Small)...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model_name = "dynamic-ik-3d"

for img_name in images:
    img_path = os.path.join("inputs", img_name)
    log(f"\nProcessing: {img_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        log(f"Could not read image: {img_name}")
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ==========================================
    # STAGE 1: 3D POSE & IK CORRECTION
    # ==========================================
    log("\n--- STAGE 1: 3D POSE & DYNAMIC IK ---")
    results = pose.process(img_rgb)
    
    calculations = {}
    
    if results.pose_world_landmarks:
        landmarks = results.pose_world_landmarks.landmark
        
        # Extract raw 3D coordinates
        raw_coords = {}
        for i, name in enumerate(keypoint_names):
            raw_coords[name] = [landmarks[i].x, landmarks[i].y, landmarks[i].z]
            calculations[name] = {
                "X_m": round(landmarks[i].x, 4), 
                "Y_m": round(landmarks[i].y, 4), 
                "Z_m": round(landmarks[i].z, 4),
                "Visibility": round(results.pose_landmarks.landmark[i].visibility, 2)
            }
        
        # Calculate dynamic bone lengths from visible torso
        thigh_len, shin_len = get_dynamic_bone_lengths(
            [raw_coords["left_shoulder"], raw_coords["right_shoulder"]],
            [raw_coords["left_hip"], raw_coords["right_hip"]]
        )
        log(f"Dynamic Bone Lengths Calculated -> Thigh: {thigh_len:.3f}m, Shin: {shin_len:.3f}m")
        
        # Apply FABRIK IK to fix hidden/occluded knees and ankles
        log("\nApplying FABRIK IK to fix occluded lower body...")
        log("-"*70)
        
        # Left Leg IK
        left_knee_corrected = fabrik_ik_solve(
            raw_coords["left_hip"], raw_coords["left_ankle"], thigh_len, shin_len
        )
        # Right Leg IK
        right_knee_corrected = fabrik_ik_solve(
            raw_coords["right_hip"], raw_coords["right_ankle"], thigh_len, shin_len
        )
        
        # Update calculations with corrected IK data
        calculations["left_knee"] = {
            "X_m": round(left_knee_corrected[0], 4), "Y_m": round(left_knee_corrected[1], 4), 
            "Z_m": round(left_knee_corrected[2], 4), "Status": "IK_CORRECTED"
        }
        calculations["right_knee"] = {
            "X_m": round(right_knee_corrected[0], 4), "Y_m": round(right_knee_corrected[1], 4), 
            "Z_m": round(right_knee_corrected[2], 4), "Status": "IK_CORRECTED"
        }
        
        # Print final results
        for name in keypoint_names:
            data = calculations[name]
            status = data.get("Status", "RAW")
            log(f"{name:<15} | X:{data['X_m']:<6.3f} | Y:{data['Y_m']:<6.3f} | Z:{data['Z_m']:<6.3f} | [{status}]")
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
    
    d_min = float(np.min(depth_map_raw))
    d_max = float(np.max(depth_map_raw))
    log(f"Depth Range: {d_min:.2f} (Closest) to {d_max:.2f} (Furthest)")
    
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
