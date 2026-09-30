import os
import glob
import sys
import numpy as np
import torch
import trimesh
import cv2
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms

INPUT_DIR = "inputs"
OUTPUT_DIR = "out_pear"
DEVICE = "cpu"

IMAGE_EXTS = ("jpg", "jpeg", "jpge", "png", "bmp", "webp", "tif", "tiff")

# Add PEAR repo to Python path
sys.path.insert(0, "PEAR")

def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files

def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.25):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox.astype(np.float32)

def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    trans = cv2.getAffineTransform(
        np.float32([[0, 0], [bb_width, 0], [0, bb_height]]),
        np.float32([[out_shape[1] * 0.5, out_shape[0] * 0.5],
                    [out_shape[1] * 0.5, out_shape[0] * 0.5],
                    [out_shape[1] * 0.5, out_shape[0] * 0.5]])
    )
    trans = cv2.getAffineTransform(
        np.float32([[bb_c_x - bb_width * 0.5, bb_c_y - bb_height * 0.5],
                    [bb_c_x + bb_width * 0.5, bb_c_y - bb_height * 0.5],
                    [bb_c_x - bb_width * 0.5, bb_c_y + bb_height * 0.5]]),
        np.float32([[0, 0], [out_shape[1], 0], [0, out_shape[0]]])
    )
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])),
                               flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = cv2.invertAffineTransform(trans)
    return img_patch, trans, inv_trans

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = find_images()
    if not images:
        print(f"[ERROR] No images in {INPUT_DIR}/")
        sys.exit(1)
    print(f"[*] Found {len(images)} image(s)")

    # ---- Load PEAR (CPU) ----
    try:
        from models.pipeline.ehm_pipeline import Ehm_Pipeline
        from models.modules.ehm import EHM_v2
        from utils.general_utils import ConfigDict, add_extra_cfgs
        from huggingface_hub import hf_hub_download
        import lightning

        meta_cfg = ConfigDict(model_config_path=os.path.join('configs', 'infer.yaml'))
        meta_cfg = add_extra_cfgs(meta_cfg)
        lightning.fabric.seed_everything(10)

        # Download model from HuggingFace
        repo_id = "BestWJH/PEAR_models"
        filename = "pear_model.pt"
        ehm_basemodel = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

        ehm_model = Ehm_Pipeline(meta_cfg)
        _state = torch.load(ehm_basemodel, map_location='cpu', weights_only=True)
        ehm_model.backbone.load_state_dict(_state['backbone'], strict=False)
        ehm_model.head.load_state_dict(_state['head'], strict=False)

        # Force CPU
        ehm_model = ehm_model.to(DEVICE)
        ehm_model.eval()

        ehm = EHM_v2("assets/FLAME", "assets/SMPLX")
        ehm = ehm.to(DEVICE)

        print("[OK] PEAR loaded on CPU")
    except Exception as e:
        print(f"[ERROR] Failed to load PEAR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load YOLO detector
    detector = YOLO('./model_zoo/yolov8x.pt')
    transform = transforms.ToTensor()

    # ---- Process Each Image ----
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[*] Processing {img_path}")
        try:
            original_img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if original_img is None:
                print(f"    [ERROR] Failed to read image")
                continue
            original_img = original_img[:, :, ::-1].copy()  # BGR to RGB
            original_img = cv2.resize(original_img,
                                      (original_img.shape[1] * 2, original_img.shape[0] * 2),
                                      interpolation=cv2.INTER_CUBIC)
            original_img_height, original_img_width = original_img.shape[:2]

            yolo_bbox = detector.predict(original_img, device='cpu', classes=0,
                                         conf=0.5, save=False, verbose=False)[0].boxes.xyxy.cpu().numpy()

            if len(yolo_bbox) == 0:
                print(f"    [ERROR] No person detected")
                continue

            for bbox_id in range(len(yolo_bbox)):
                yolo_bbox_xywh = np.zeros((4))
                yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
                yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
                yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
                yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

                bbox = process_bbox(yolo_bbox_xywh, original_img_width, original_img_height,
                                    [256, 256], ratio=1.25)
                if bbox is None:
                    continue

                img_patch, trans, inv_trans = generate_patch_image(
                    cvimg=original_img, bbox=bbox, scale=1.0, rot=0.0,
                    do_flip=False, out_shape=[256, 256])
                img_patch = transform(img_patch.astype(np.float32)) / 255
                img_patch = img_patch.unsqueeze(0).to(DEVICE)

                outputs = ehm_model(img_patch)
                pd_smplx_dict = ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')

                # Export mesh
                vertices = pd_smplx_dict['vertices'][0].detach().cpu().numpy()
                faces = pd_smplx_dict['faces'].detach().cpu().numpy() if 'faces' in pd_smplx_dict else ehm.smplx.faces

                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                out_obj = os.path.join(OUTPUT_DIR, f"{name}_pear_{bbox_id}.obj")
                mesh.export(out_obj)
                print(f"    [OK] Exported {out_obj}")

        except Exception as e:
            print(f"    [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
