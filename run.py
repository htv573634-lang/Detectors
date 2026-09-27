import os
import cv2
import mediapipe as mp
import json
from datetime import datetime

# ==========================================
# 1. AUTO-SETUP FOLDERS
# ==========================================
os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

print("="*60)
print("🚀 SIMPLE 3D DETECTOR RUNNER")
print("="*60)

# Check if there are images to process
input_images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not input_images:
    print("⚠️ No images found in the 'inputs/' folder.")
    print("👉 Please put an image (e.g., 'person.jpg') in the 'inputs/' folder and run again.")
    exit()

# ==========================================
# 2. INITIALIZE DETECTOR (Auto-downloads models on first run)
# ==========================================
print("\n⚙️ Loading MediaPipe Pose (will auto-download weights if needed)...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 3. PROCESS EACH IMAGE
# ==========================================
for img_name in input_images:
    img_path = os.path.join("inputs", img_name)
    print(f"\n📸 Processing: {img_name}")
    
    # Load image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run calculation
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        print(f"   ❌ No human detected in {img_name}. Skipping.")
        continue
        
    print("   ✅ Human detected! Calculating 3D coordinates...")
    
    # Extract raw math for key body parts
    calculations = {}
    key_points = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    
    print("   📊 RAW CALCULATIONS (X=Left/Right, Y=Top/Bottom, Z=Depth):")
    print("   " + "-"*55)
    
    for point in key_points:
        lm = results.pose_landmarks.landmark[point.value]
        name = point.name
        
        # Store data
        calculations[name] = {
            "X": round(lm.x, 4),
            "Y": round(lm.y, 4),
            "Z": round(lm.z, 4),          # Lower Z = Closer to camera
            "Visibility": round(lm.visibility, 2)
        }
        
        # Print to console log
        print(f"   {name:<15} | X: {lm.x:<6.4f} | Y: {lm.y:<6.4f} | Z: {lm.z:<6.4f}")
    
    print("   " + "-"*55)
    
    # ==========================================
    # 4. SAVE ARTIFACTS
    # ==========================================
    base_name = os.path.splitext(img_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # A. Save Visual Proof (Image with skeleton drawn on it)
    output_img_name = f"{base_name}_result_{timestamp}.jpg"
    output_img_path = os.path.join("artifacts", output_img_name)
    
    # Draw landmarks
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    cv2.imwrite(output_img_path, img)
    print(f"   💾 Saved visual proof to: artifacts/{output_img_name}")
    
    # B. Save Raw Calculation Data (JSON)
    output_json_name = f"{base_name}_data_{timestamp}.json"
    output_json_path = os.path.join("artifacts", output_json_name)
    
    with open(output_json_path, "w") as f:
        json.dump(calculations, f, indent=4)
    print(f"   💾 Saved raw math data to: artifacts/{output_json_name}")

print("\n" + "="*60)
print("🎉 ALL DONE! Check the 'artifacts/' folder for your results.")
print("="*60)

# Cleanup
pose.close()
