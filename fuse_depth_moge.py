import os
import numpy as np
import cv2
import trimesh
import open3d as o3d

OUTPUT_DIR = "out_moge"

# ── CALIBRATION ──
SUBDIVIDE = False          # At res=9, grid is already ultra-dense
SMOOTH_ITERS = 6           # Light smoothing to remove grid jitter
DENSITY_TRIM = 0.01        # Remove 1% lowest-density triangles


def find_name():
    metas = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_meta.txt")]
    if not metas:
        return "test-2"
    metas.sort(key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
    return metas[0].replace("_meta.txt", "")


def triangulate(points, mask):
    """Build mesh from MoGe grid by connecting adjacent pixels."""
    H, W = mask.shape
    idx = -np.ones((H, W), dtype=np.int32)
    v = mask > 0.5
    verts = points[v]
    idx[v] = np.arange(len(verts))

    print(f"    Building triangles from {len(verts)} vertices...")
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = idx[i, j]
            b = idx[i + 1, j]
            c = idx[i, j + 1]
            d = idx[i + 1, j + 1]
            if a >= 0 and b >= 0 and c >= 0:
                faces.append([a, b, c])
            if b >= 0 and d >= 0 and c >= 0:
                faces.append([b, d, c])

    faces = np.array(faces, dtype=np.int32)
    print(f"    Triangles: {len(faces)}")
    return verts, faces


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    NAME = find_name()
    print(f"[*] Processing: {NAME}")

    # Load MoGe raw grid
    points = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_points_raw.npy"))
    mask = np.load(os.path.join(OUTPUT_DIR, f"{NAME}_mask_raw.npy"))
    H_m, W_m = mask.shape
    print(f"[*] MoGe grid: {W_m}x{H_m} = {H_m*W_m} cells")

    # Clean up NaNs
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    # Triangulate directly from the grid (no Poisson — preserves topology)
    print("[*] Triangulating MoGe grid...")
    verts, faces = triangulate(points, mask)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    print(f"    Raw mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Convert to Open3D for cleanup + smoothing
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )

    print("[*] Removing degenerate + duplicate geometry...")
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_unreferenced_vertices()
    print(f"    After cleanup: {len(o3d_mesh.vertices)} verts, {len(o3d_mesh.triangles)} faces")

    if SUBDIVIDE:
        print("[*] Midpoint subdivision...")
        o3d_mesh = o3d_mesh.subdivide_midpoint(number_of_iterations=1)
        print(f"    After subdivision: {len(o3d_mesh.vertices)} verts")

    print(f"[*] Taubin smoothing ({SMOOTH_ITERS} iterations)...")
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERS)
    o3d_mesh.compute_vertex_normals()

    # Export
    final = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )

    obj_path = os.path.join(OUTPUT_DIR, f"{NAME}_mesh_fused.obj")
    glb_path = os.path.join(OUTPUT_DIR, f"{NAME}_mesh_fused.glb")
    final.export(obj_path)
    final.export(glb_path)
    print(f"[OK] {obj_path}")
    print(f"[OK] {glb_path}")
    print(f"    Size: {os.path.getsize(glb_path)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
