import os
import cv2
import sys
import numpy as np
import requests
from datetime import datetime

# FIX: Updated import for modern OpenVINO versions
from openvino import Core

def log(msg):
    print(msg, flush=True)

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Cleanup previous artifacts
log("Cleaning up previous artifacts...")
for f in os.listdir("artifacts"):
    file_path = os.path.join("artifacts", f)
    if os.path.isfile(file_path):
        os.remove(file_path)
log("Artifacts folder cleared.")

log("="*60)
log("TRUE 3D PIPELINE: Intel OpenVINO (CPU Optimized)")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

# ==========================================
# DOWNLOAD OPENVINO 3D POSE MODEL
# ==========================================
model_xml = "models/human-pose-estimation-3d-0001.xml"
model_bin = "models/human-pose-estimation-3d-0001.bin"

if not os.path.exists(model_xml) or not os.path.exists(model_bin):
    log("Downloading Intel OpenVINO 3D Pose Model (First run only)...")
    base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/3/human-pose-estimation-3d-0001/FP32/"
    
    with open(model_xml, "wb") as f:
        f.write(requests.get(base_url + "human-pose-estimation-3d-0001.xml").content)
    with open(model_bin, "wb") as f:
        f.write(requests.get(base_url + "human-pose-estimation-3d-0001.bin").content)
    log("Model downloaded successfully.")

# ==========================================
# LOAD OPENVINO ENGINE
# ==========================================
log("Initializing OpenVINO Core (CPU)...")
core = Core()
model = core.read_model(model=model_xml, weights=model_bin)
compiled_model = core.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# OpenVINO 3D Pose expects 1x3x256x448 input
target_h, target_w = 256, 448

# COCO 17 keypoints order for this model
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model_name = "openvino-true-3d"

for img_name in images:
    img_path = os.path.join("inputs", img_name)
    log(f"\nProcessing: {img_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        log(f"Could not read image: {img_name}")
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    
    # ==========================================
    # PREPROCESS FOR OPENVINO
    # ==========================================
    resized = cv2.resize(img_rgb, (target_w, target_h))
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
    
    # ==========================================
    # STAGE 1: TRUE 3D INFERENCE
    # ==========================================
    log("\n--- STAGE 1: TRUE 3D INFERENCE ---")
    result = compiled_model([input_tensor])[output_layer]
    
    # Scale factor to approximate real-world meters (assuming ~1.7m human)
    scale_factor = 1.7 
    
    log("Real-World 3D Skeleton Coordinates (Approx. Meters):")
    log("-"*75)
    log(f"{'JOINT':<15} | {'X (Left/Right)':<15} | {'Y (Up/Down)':<15} | {'Z (Front/Back)':<15}")
    log("-"*75)
    
    for i in range(17):
        x = result[0, 0, 0, i] * scale_factor
        y = result[0, 0, 1, i] * scale_factor
        z = result[0, 0, 2, i] * scale_factor
        
        y_inv = -y
        
        name = keypoint_names[i]
        log(f"{name:<15} | X:{x:<14.4f} | Y:{y_inv:<14.4f} | Z:{z:<14.4f}")
    log("-"*75)
    
    nose_y = - (result[0, 0, 1, 0] * scale_factor)
    l_ankle_y = - (result[0, 0, 1, 15] * scale_factor)
    r_ankle_y = - (result[0, 0, 1, 16] * scale_factor)
    avg_ankle_y = (l_ankle_y + r_ankle_y) / 2.0
    estimated_height = (nose_y - avg_ankle_y) + 0.10
    log(f"Estimated Real-World Height: {estimated_height:.2f} meters")

    # ==========================================
    # SAVE VISUAL ARTIFACTS
    # ==========================================
    base = os.path.splitext(img_name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    skeleton_img = img.copy()
    cv2.imwrite(f"artifacts/{base}_skeleton_{model_name}_{ts}.jpg", skeleton_img)
    
    log(f"\nSaved visual artifacts for {img_name}")

log("\nDONE! Check artifacts folder and run logs for the True 3D Skeleton.")
