import os
import sys
import glob
import json
import time
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import open3d as o3d

# ── CONFIG ──
INPUT_DIR = "inputs"
OUTPUT_DIR = "out_fusion"
IMAGE_EXTS = (
    "jpg", "jpeg", "jpge", "jpe", "jfif",
    "png", "bmp", "webp", "tif", "tiff", "gif", "ppm"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_images():
    files = []
    for ext in IMAGE_EXTS:
        for pat in (f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"):
            files.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    return sorted(set(f for f in files if not os.path.basename(f).startswith(".")))


def points_to_glb(points, colors, out_path, name):
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid] if colors is not None else None
    if len(points) == 0:
        log(f"    {name}: no valid points")
        return

    if len(points) > 500_000:
        idx = np.random.choice(len(points), 500_000, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None

    scene = trimesh.Scene()
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    scene.add_geometry(pcd)
    scene.export(out_path)
    size_kb = os.path.getsize(out_path) / 1024
    log(f"    {name}: {len(points)} pts -> {out_path} ({size_kb:.1f} KB)")


def poisson_mesh_from_points(points, out_path, name, depth=9):
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) < 1000:
        return
    if len(points) > 200_000:
        idx = np.random.choice(len(points), 200_000, replace=False)
        points = points[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    to_remove = densities < np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(to_remove)
    mesh.compute_vertex_normals()
    mesh_path = out_path.replace(".glb", "_mesh.glb")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    log(f"    {name} mesh: {len(mesh.vertices)} verts -> {mesh_path}")


# ══════════════════════════════════════════════════════════════
# DepthPro
# ══════════════════════════════════════════════════════════════
def run_depthpro(img_rgb, out_dir, base_name):
    log("[DepthPro] Running...")
    try:
        import depth_pro
        device = torch.device("cpu")
        model, transform = depth_pro.create_model_and_transforms()
        model.eval().to(device)

        img_pil = Image.fromarray(img_rgb)
        image_t = transform(img_pil).to(device)

        with torch.no_grad():
            prediction = model.infer(image_t)

        depth = prediction["depth"].cpu().numpy().squeeze()
        H, W = depth.shape
        f_px = prediction.get("focallength_px", None)
        if f_px is not None:
            f_px = f_px.item()
        else:
            f_px = W * 0.8

        cx, cy = W / 2.0, H / 2.0
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        x3d = (u - cx) * z / f_px
        y3d = (v - cy) * z / f_px
        points = np.stack([x3d, y3d, z], axis=-1).reshape(-1, 3)
        colors = img_rgb.reshape(-1, 3)

        valid = np.isfinite(points).all(axis=1) & (z.reshape(-1) > 0)
        valid_pts = points[valid]
        valid_colors = colors[valid]

        glb_path = os.path.join(out_dir, f"{base_name}_depthpro.glb")
        points_to_glb(valid_pts, valid_colors, glb_path, "DepthPro")
        poisson_mesh_from_points(valid_pts, glb_path, "DepthPro")

        depth_valid = depth[np.isfinite(depth)]
        return {"model": "DepthPro", "status": "ok",
                "depth_min": float(depth_valid.min()),
                "depth_max": float(depth_valid.max()),
                "valid_points": int(valid_pts.shape[0]),
                "glb": glb_path}
    except Exception as e:
        log(f"[DepthPro] FAILED: {e}")
        import traceback; traceback.print_exc()
        return {"model": "DepthPro", "status": "error", "reason": str(e)}


# ══════════════════════════════════════════════════════════════
# UniDepth V2
# ══════════════════════════════════════════════════════════════
def run_unidepth(img_rgb, out_dir, base_name):
    log("[UniDepth V2] Running...")
    try:
        from unidepth.models import UniDepthV2
        device = torch.device("cpu")
        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14").to(device).eval()

        rgb_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        with torch.no_grad():
            preds = model.infer(rgb_t)

        points = preds["points"].squeeze(0).cpu().numpy()
        depth = preds["depth"].squeeze(0).cpu().numpy()

        H, W = depth.shape[:2]
        pts_flat = points.reshape(-1, 3)
        colors = img_rgb.reshape(-1, 3)

        valid = np.isfinite(pts_flat).all(axis=1) & (np.abs(pts_flat[:, 2]) > 1e-6)
        valid_pts = pts_flat[valid]
        valid_colors = colors[valid]

        glb_path = os.path.join(out_dir, f"{base_name}_unidepth.glb")
        points_to_glb(valid_pts, valid_colors, glb_path, "UniDepth V2")
        poisson_mesh_from_points(valid_pts, glb_path, "UniDepth V2")

        depth_valid = depth[np.isfinite(depth)]
        return {"model": "UniDepth V2", "status": "ok",
                "depth_min": float(depth_valid.min()),
                "depth_max": float(depth_valid.max()),
                "valid_points": int(valid_pts.shape[0]),
                "glb": glb_path}
    except Exception as e:
        log(f"[UniDepth V2] FAILED: {e}")
        import traceback; traceback.print_exc()
        return {"model": "UniDepth V2", "status": "error", "reason": str(e)}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    images = find_images()
    if not images:
        log(f"No images found in {INPUT_DIR}/")
        sys.exit(1)

    log(f"Found {len(images)} image(s)")

    all_results = {}
    for i, img_path in enumerate(images, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        log(f"\n[{i}/{len(images)}] {img_path}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            log(f"  Cannot read image")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        image_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(image_dir, exist_ok=True)

        results = [
            run_depthpro(img_rgb, image_dir, base_name),
            run_unidepth(img_rgb, image_dir, base_name),
        ]

        all_results[base_name] = {
            "source_image": os.path.basename(img_path),
            "results": results,
            "best_by_dense_points": max(
                (r for r in results if r.get("status") == "ok"),
                key=lambda r: r.get("valid_points", 0),
                default={"model": "none"},
            )["model"],
        }

    index_path = os.path.join(OUTPUT_DIR, "comparison_index.json")
    with open(index_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log(f"\n[SUCCESS] All models processed. Index: {index_path}")
    for name, data in all_results.items():
        log(f"  {name}:")
        for r in data["results"]:
            status = r.get("status", "?")
            if status == "ok":
                log(f"    {r['model']}: {r['valid_points']} pts, "
                    f"depth {r['depth_min']:.3f}-{r['depth_max']:.3f} m")
            else:
                log(f"    {r['model']}: FAILED ({r.get('reason', 'unknown')})")


if __name__ == "__main__":
    main()
