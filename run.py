import os
import cv2
import json
import sys
from datetime import datetime
from ultralytics import YOLO

def log(msg):
    print(msg, flush=True)

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

log("="*60)
log("3D DETECTOR RUNNING (YOLO11-X - Extra Large / Occlusion Master)")
log("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]

if not images:
    log("No images found in inputs/ folder.")
    sys.exit(0)

log("Loading YOLO11-X Pose (Extra Large - Best for crossed arms)...")
# Using the 'x' (Extra Large) model for maximum accuracy on hidden limbs
model = YOLO('yolo11x-pose.pt') 

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model_name = "yolo11x-pose"

for img_name in images:
    img_path = os.path.join("inputs", img_name)
    log(f"\nProcessing: {img_name}")
    
    results = model(img_path)
    
    if not results[0].keypoints or results[0].keypoints.xy is None:
        log(f"No human detected in {img_name}")
        continue
    
    log("Human detected! Calculating...")
    
    kpts = results[0].keypoints.xy[0]
    confs = results[0].keypoints.conf[0]
    
    calculations = {}
    
    log("RAW CALCULATIONS (X=Left/Right, Y=Top/Bottom, Conf=Confidence):")
    log("-"*65)
    
    for i, name in enumerate(keypoint_names):
        x = round(float(kpts[i][0]), 4)
        y = round(float(kpts[i][1]), 4)
        conf = round(float(confs[i]), 4)
        
        calculations[name] = {
            "X": x,
            "Y": y,
            "Confidence": conf
        }
        
        status = ""
        if conf < 0.50:
            status = " [️ LOW CONFIDENCE]"
            
        log(f"{name:<15} | X:{x:<7.4f} | Y:{y:<7.4f} | Conf:{conf}{status}")
    
    log("-"*65)
    
    base = os.path.splitext(img_name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    img_out = f"artifacts/{base}_result_{model_name}_{ts}.jpg"
    json_out = f"artifacts/{base}_data_{model_name}_{ts}.json"
    
    annotated_frame = results[0].plot()
    cv2.imwrite(img_out, annotated_frame)
    
    with open(json_out, "w") as f:
        json.dump(calculations, f, indent=4)
    
    log(f"Saved to artifacts/ as {base}_*_{model_name}_*")

log("\nDONE! Check artifacts.")
