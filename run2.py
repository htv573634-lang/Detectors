import os
import json
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from transformers import pipeline

INPUT_DIR = "inputs2"
OUTPUT_DIR = "artifacts2"
IMAGE_NAME = "test.jpge"          # your exact filename
IMAGE_PATH = os.path.join(INPUT_DIR, IMAGE_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load image ----------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =========================================================
# STAGE 1 : MediaPipe Pose  -> 3D keypoints
# =========================================================
print("--- STAGE 1: 3D ANATOMICAL SKELETON ---")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

results = pose.process(img_rgb)
pose.close()

if results.pose_world_landmarks is None:
    raise RuntimeError("No pose detected in the image.")

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

pose_data = {}
for name, lm in zip(KEYPOINT_NAMES, results.pose_world_landmarks.landmark):
    pose_data[name] = {"X": lm.x, "Y": lm.y, "Z": lm.z}
    print(f"{name:15s} | X:{lm.x:.4f} | Y:{lm.y:.4f} | Z:{lm.z:.4f}")

pose_out = os.path.join(OUTPUT_DIR, "mediapipe_test.json")
with open(pose_out, "w") as f:
    json.dump(pose_data, f, indent=2)
print(f"Saved: {pose_out}")

# =========================================================
# STAGE 2 : Depth Anything V2 Small -> depth map
# =========================================================
print("\n--- STAGE 2: DEPTH MAP CALCULATION ---")

depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)

img_pil = Image.fromarray(img_rgb)              # fixes the TypeError
depth_result = depth_pipe(img_pil)

if isinstance(depth_result, list):
    depth_result = depth_result[0]

if "depth" in depth_result:
    depth_map = np.array(depth_result["depth"])
elif "predicted_depth" in depth_result:
    depth_map = depth_result["predicted_depth"].squeeze().cpu().numpy()
else:
    raise KeyError(f"Unexpected depth output keys: {depth_result.keys()}")

d = depth_map.astype(np.float32)
d = (d - d.min()) / (d.max() - d.min() + 1e-8)
d_uint8 = (d * 255).astype(np.uint8)

depth_out = os.path.join(OUTPUT_DIR, "depthanything_test.png")
cv2.imwrite(depth_out, d_uint8)

print(f"Minimum Depth : {d_uint8.min():.2f}")
print(f"Maximum Depth : {d_uint8.max():.2f}")
print(f"Average Depth : {d_uint8.mean():.2f}")
print(f"Saved: {depth_out}")

print("\nDONE. Results in artifacts2/")
