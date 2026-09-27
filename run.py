import os
import cv2
import mediapipe as mp
import json
from datetime import datetime

os.makedirs("inputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

print("="*60)
print("3D DETECTOR RUNNING")
print("="*60)

images = [f for f in os.listdir("inputs") if f.lower().endswith(('.png','.jpg','.jpeg'))]

if not images:
    print("No images found in inputs/ folder.")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

for img_name in images:
    img_path = os.path.join("inputs", img_name)
    print(f"\nProcessing: {img_name}")
    
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        print(f"No human detected in {img_name}")
        continue
    
    print("Human detected! Calculating...")
    
    calculations = {}
    key_points = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    
    print("RAW CALCULATIONS (X=Left/Right, Y=Top/Bottom, Z=Depth):")
    print("-"*55)
    
    for point in key_points:
        lm = results.pose_landmarks.landmark[point.value]
        name = point.name
        calculations[name] = {
            "X": round(lm.x, 4),
            "Y": round(lm.y, 4),
            "Z": round(lm.z, 4),
            "Visibility": round(lm.visibility, 2)
        }
        print(f"{name:<15} | X:{lm.x:<7.4f} | Y:{lm.y:<7.4f} | Z:{lm.z:<7.4f}")
    
    print("-"*55)
    
    base = os.path.splitext(img_name)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
    )
    cv2.imwrite(f"artifacts/{base}_result_{ts}.jpg", img)
    
    with open(f"artifacts/{base}_data_{ts}.json", "w") as f:
        json.dump(calculations, f, indent=4)
    
    print(f"Saved to artifacts/")

pose.close()
print("\nDONE! Check artifacts.")
