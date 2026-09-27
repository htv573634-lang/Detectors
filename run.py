import os
import cv2
import sys
import numpy as np
import torch
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

def log(msg):
    print(msg, flush=True)

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Cleanup previous artifacts
log("Cleaning up previous artifacts...")
for f in os.listdir("artifacts"):
    file_path = os.path.join("artifacts", f)
    if os.path.isfile(file_path):
        os.remove(file_path)
log("Artifacts folder cleared.")

log("="*60)
log("ULTRA CLASS 3D PIPELINE: YOLO11 + VideoPose3D (Meta)")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

# ==========================================
# LOAD MODELS
# ==========================================
log("Loading YOLO11-Large Pose (2D Accuracy)...")
model = YOLO('yolo11l-pose.pt') 

log("Loading Depth Anything V2 (Small)...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

log("Loading Ultra Class 3D Lifter (VideoPose3D - CPU)...")
# This downloads the Meta AI model that lifts 2D to 3D using Transformers
try:
    model_3d = torch.hub.load('facebookresearch/video_pose_3d', 'pose3d', pretrained=True)
    model_3d.eval()
    # Force CPU
    device = torch.device('cpu')
    model_3d.to(device)
    log("3D Lifter loaded successfully.")
except Exception as e:
    log(f"Warning: Could not load 3D lifter. Error: {e}")
    log("Falling back to 2D only.")
    model_3d = None

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model_name = "ultra-class-3d"

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
    # STAGE 1: YOLO 2D POSE
    # ==========================================
    log("\n--- STAGE 1: YOLO 2D POSE ---")
    results = model(img_path, augment=True)
    
    if not (results[0].keypoints and results[0].keypoints.xy is not None):
        log("No human detected by YOLO.")
        continue
        
    kpts = results[0].keypoints.xy[0]
    confs = results[0].keypoints.conf[0]
    
    # ==========================================
    # STAGE 2: ULTRA CLASS 3D LIFTING
    # ==========================================
    log("\n--- STAGE 2: 3D LIFTING (Meta VideoPose3D) ---")
    
    if model_3d is not None:
        # Prepare 2D keypoints for the 3D lifter
        # The lifter expects normalized coordinates centered on the image
        kpts_2d = kpts.numpy()
        
        # Normalize: Center the pose and scale to [-1, 1]
        kpts_2d[:, 0] = (kpts_2d[:, 0] - w / 2) / w
        kpts_2d[:, 1] = (kpts_2d[:, 1] - h / 2) / h
        
        # Add batch and time dimensions: (1, 1, 17, 2)
        input_2d = torch.tensor(kpts_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Run the 3D Lifter
        with torch.no_grad():
            output_3d = model_3d(input_2d)
            
        # Output is in millimeters, relative to the root (hips). Convert to meters.
        pose_3d_meters = output_3d[0, 0].numpy() / 1000.0
        
        log("Real-World 3D Skeleton (Meters, Rooted at Hips):")
        log("-"*75)
        log(f"{'JOINT':<15} | {'X (Left/Right)':<15} | {'Y (Up/Down)':<15} | {'Z (Front/Back)':<15}")
        log("-"*75)
        
        for i, name in enumerate(keypoint_names):
            x, y, z = pose_3d_meters[i]
            # Invert Y so Up is Positive
            y_inv = -y 
            conf = float(confs[i])
            status = "[VISIBLE]" if conf > 0.50 else "[OCCLUDED]"
            log(f"{name:<15} | X:{x:<14.4f} | Y:{y_inv:<14.4f} | Z:{z:<14.4f} {status}")
        log("-"*75)
        
        # Calculate Height
        nose_y = -pose_3d_meters[0][1]
        l_ankle_y = -pose_3d_meters[15][1]
        r_ankle_y = -pose_3d_meters[16][1]
        avg_ankle_y = (l_ankle_y + r_ankle_y) / 2.0
        estimated_height = (nose_y - avg_ankle_y) + 0.10
        log(f"Estimated Real-World Height: {estimated_height:.2f} meters")
    else:
        log("3D Lifter not available. Skipping 3D calculation.")

    # ==========================================
    # SAVE VISUAL ARTIFACTS
    # ==========================================
    base = os.path.splitext(img_name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save YOLO Skeleton
    if results[0].keypoints:
        annotated = results[0].plot()
        cv2.imwrite(f"artifacts/{base}_skeleton_{model_name}_{ts}.jpg", annotated)
    
    log(f"\nSaved visual artifacts for {img_name}")

log("\nDONE! Check artifacts folder and run logs for the Ultra Class 3D Skeleton.")
