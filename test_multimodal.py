import os
import cv2
import torch
import gc
import numpy as np
from PIL import Image
import mediapipe as mp
from insightface.app import FaceAnalysis
from transformers import pipeline
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# ==========================================
# CONFIGURATION (Hardcoded for CI)
# ==========================================
CONFIG = {
    "input_dir": "inputs",
    "output_dir": "data/output",
    "models": {
        "depth": {
            "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
            "device": "cpu"
        },
        "face": {
            "model_name": "buffalo_l",
            "device": "cpu"
        },
        "smpl": {
            "model_path": "data/smpl/SMPL_NEUTRAL.pkl",
            "device": "cpu"
        },
        "zero123": {
            "model_id": "sudo-ai/zero123plus-v1.2",
            "custom_pipeline": "sudo-ai/zero123plus-pipeline",
            "num_inference_steps": 28,
            "device": "cpu"
        }
    }
}

def setup_output_directories(output_dir):
    """Creates the output directory and its subfolders if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "face"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "smpl"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "views"), exist_ok=True)
    return output_dir

def free_memory():
    """Force garbage collection to prevent Out Of Memory (OOM) on GitHub runners."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_single_image(input_path, output_dir):
    print(f"\n[*] Processing: {os.path.basename(input_path)}")
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f"    -> [ERROR] Could not read image: {input_path}")
        return
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # ==========================================
    # 1. FACE DETECTION (InsightFace)
    # ==========================================
    print("    -> Running Face Detection...")
    try:
        face_app = FaceAnalysis(name=CONFIG["models"]["face"]["model_name"], providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1)
        faces = face_app.get(image_rgb)
        face_img = image_bgr.copy()
        if faces:
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(face_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                face_crop = image_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if face_crop.size > 0:
                    Image.fromarray(face_crop).save(os.path.join(output_dir, "face", "face_crop.jpg"))
        cv2.imwrite(os.path.join(output_dir, "face", "face_output.jpg"), face_img)
        print(f"        Detected {len(faces)} face(es)")
    except Exception as e:
        print(f"        [WARNING] Face detection failed: {e}")
    free_memory()

    # ==========================================
    # 2. SMPL MESH GENERATION (smplx + MediaPipe Proxy)
    # ==========================================
    print("    -> Checking SMPL Model...")
    smpl_path = CONFIG["models"]["smpl"]["model_path"]
    if os.path.exists(smpl_path):
        try:
            import smplx
            smpl_model = smplx.create(model_path=smpl_path, model_type='smpl', gender='neutral', batch_size=1)
            
            mp_pose = mp.solutions.pose
            pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
            pose_results = pose_detector.process(image_rgb)
            
            if pose_results.pose_landmarks:
                print("        Generating SMPL Mesh...")
                output = smpl_model()
                smpl_model.save_obj(output, os.path.join(output_dir, "smpl", "smpl_mesh.obj"))
                print("        SMPL mesh saved.")
            else:
                print("        [WARNING] No pose landmarks detected for SMPL.")
        except Exception as e:
            print(f"        [WARNING] SMPL generation failed: {e}")
    else:
        print(f"        [SKIP] SMPL model not found at {smpl_path}. Skipping SMPL generation.")
    free_memory()

    # ==========================================
    # 3. DEPTH ESTIMATION (Depth Anything V2)
    # ==========================================
    print("    -> Running Depth Estimation...")
    try:
        depth_pipe = pipeline(task="depth-estimation", model=CONFIG["models"]["depth"]["model_id"], device=CONFIG["models"]["depth"]["device"])
        depth_result = depth_pipe(image_pil)
        depth_result["depth"].save(os.path.join(output_dir, "depth", "depth_map.png"))
        print("        Depth map saved.")
        del depth_pipe
    except Exception as e:
        print(f"        [WARNING] Depth estimation failed: {e}")
    free_memory()

    # ==========================================
    # 4. MULTI-VIEW GENERATION (Zero123++)
    # ==========================================
    print("    -> Generating 6-View Grid (This is the slowest step)...")
    try:
        zero123_pipe = DiffusionPipeline.from_pretrained(
            CONFIG["models"]["zero123"]["model_id"],
            custom_pipeline=CONFIG["models"]["zero123"]["custom_pipeline"],
            torch_dtype=torch.float32
        )
        zero123_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            zero123_pipe.scheduler.config, timestep_spacing='trailing'
        )
        zero123_pipe.to(CONFIG["models"]["zero123"]["device"])

        input_square = image_pil.resize((512, 512), Image.Resampling.LANCZOS)
        multiview_grid = zero123_pipe(input_square, num_inference_steps=CONFIG["models"]["zero123"]["num_inference_steps"]).images[0]
        multiview_grid.save(os.path.join(output_dir, "views", "multiview_grid.png"))

        # Crop the 2x3 grid into 6 separate views
        w, h = multiview_grid.size
        views = {
            'front': (0, 0, w//3, h//2), 'right': (w//3, 0, 2*w//3, h//2),
            'back': (2*w//3, 0, w, h//2), 'left': (0, h//2, w//3, h),
            'top': (w//3, h//2, 2*w//3, h), 'bottom': (2*w//3, h//2, w, h)
        }
        for name, box in views.items():
            multiview_grid.crop(box).save(os.path.join(output_dir, "views", f"{name}.png"))
        print("        6 Views generated successfully.")
        del zero123_pipe
    except Exception as e:
        print(f"        [ERROR] Zero123++ failed (Likely OOM): {e}")
    free_memory()

def main():
    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"[!] Created input directory: {input_dir}. Please add an image and run again.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"[!] No images found in {input_dir}. Please add an image and run again.")
        return

    print(f"[*] Found {len(image_files)} image(s) to process.")
    setup_output_directories(output_dir)

    for i, filename in enumerate(image_files):
        print(f"\n--- Processing Image {i+1}/{len(image_files)} ---")
        process_single_image(os.path.join(input_dir, filename), output_dir)

    print(f"\n[SUCCESS] Processing complete! Check outputs in: {output_dir}")

if __name__ == "__main__":
    main()
