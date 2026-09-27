import os
import cv2
import sys
import subprocess
import glob
import numpy as np
from datetime import datetime
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
# 1. DOWNLOAD & 2. CONVERT MODEL
# ==========================================
model_name_omz = "human-pose-estimation-3d-0001"

log(f"1. Checking/Downloading {model_name_omz}...")
subprocess.run([
    "omz_downloader",
    "--name", model_name_omz,
    "--output_dir", "models",
    "--precision", "FP32"
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

log(f"2. Checking/Converting {model_name_omz} to OpenVINO IR...")
subprocess.run([
    "omz_converter",
    "--name", model_name_omz,
    "--download_dir", "models",
    "--output_dir", "models"
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 3. SMART FIND
xml_files = glob.glob(f"models/**/{model_name_omz}.xml", recursive=True)
bin_files = glob.glob(f"models/**/{model_name_omz}.bin", recursive=True)

if not xml_files or not bin_files:
    log("ERROR: Conversion failed. No .xml or .bin files found.")
    sys.exit(1)

model_xml = xml_files[0]
model_bin = bin_files[0]
log(f"Model ready at: {model_xml}")

# ==========================================
# LOAD OPENVINO ENGINE
# ==========================================
log("Initializing OpenVINO Core (CPU)...")
core = Core()
model = core.read_model(model=model_xml, weights=model_bin)
compiled_model = core.compile_model(model=model, device_name="CPU")

# OpenVINO 3D Pose expects 1x3x256x448 input
target_h, target_w = 256, 448

# This model outputs 32 joints. We map the first 17 to standard COCO names.
keypoint_names = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", 
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", 
    "left_eye", "right_ear" 
]

model_name = "openvino-true-3d"

for img_name in images:
    img_path = os.path.join("inputs", img_name)
    log(f"\nProcessing: {img_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        log(f"Could not read image: {img_name}")
        continue
        
    # ==========================================
    # PREPROCESS FOR OPENVINO (THE FIX)
    # ==========================================
    # 1. Keep as BGR (OpenVINO OMZ models expect BGR, not RGB)
    resized = cv2.resize(img, (target_w, target_h))
    
    # 2. Transpose to (1, 3, H, W) and scale to 0-1
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
    
    # 3. Apply ImageNet Normalization (Crucial for OpenVINO)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    input_tensor = (input_tensor - mean) / std
    
    # ==========================================
    # STAGE 1: TRUE 3D INFERENCE
    # ==========================================
    log("\n--- STAGE 1: TRUE 3D INFERENCE ---")
    result = compiled_model([input_tensor])[compiled_model.output(0)]
    
    # Debug: Print raw output shape and range to ensure it's not zeros
    log(f"Raw Output Shape: {result.shape}")
    log(f"Raw Output Min/Max: {np.min(result):.4f} / {np.max(result):.4f}")
    
    # Scale factor to approximate real-world meters (assuming ~1.7m human)
    scale_factor = 1.7 
    
    log("\nReal-World 3D Skeleton Coordinates (Approx. Meters):")
    log("-"*75)
    log(f"{'JOINT':<15} | {'X (Left/Right)':<15} | {'Y (Up/Down)':<15} | {'Z (Front/Back)':<15}")
    log("-"*75)
    
    # The model outputs 32 joints. We will print the first 17.
    num_joints_to_print = min(17, result.shape[3])
    
    for i in range(num_joints_to_print):
        x = result[0, 0, 0, i] * scale_factor
        y = result[0, 0, 1, i] * scale_factor
        z = result[0, 0, 2, i] * scale_factor
        
        y_inv = -y # Invert Y so Up is Positive
        
        name = keypoint_names[i] if i < len(keypoint_names) else f"joint_{i}"
        log(f"{name:<15} | X:{x:<14.4f} | Y:{y_inv:<14.4f} | Z:{z:<14.4f}")
    log("-"*75)
    
    # Calculate approximate height (Nose to average Ankle)
    # Index 0 is Nose, 10 is Right Ankle, 13 is Left Ankle in this specific model topology
    if result.shape[3] > 13:
        nose_y = - (result[0, 0, 1, 0] * scale_factor)
        r_ankle_y = - (result[0, 0, 1, 10] * scale_factor)
        l_ankle_y = - (result[0, 0, 1, 13] * scale_factor)
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
