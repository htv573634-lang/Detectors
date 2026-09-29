import os
import numpy as np
import trimesh

RAW_VERTICES = "out_sam/mesh.vertices"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"

def load_raw_vertices(path):
    verts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    return np.array(verts, dtype=np.float64)

def poisson_reconstruct(verts):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    print("[*] Running Poisson reconstruction (depth=9)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    # Crop to original point cloud bbox to remove padding
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    print(f"[*] Poisson: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
    )

def smooth_mesh():
    if not os.path.isfile(RAW_VERTICES):
        print(f"[ERROR] {RAW_VERTICES} not found")
        return

    print(f"[*] Loading raw vertices from {RAW_VERTICES}")
    verts = load_raw_vertices(RAW_VERTICES)
    print(f"[*] Loaded {len(verts)} vertices")

    tm = poisson_reconstruct(verts)

    # Handle Scene vs Trimesh
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    print(f"[*] Input mesh: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    # Taubin smoothing
    trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=0.53, iterations=10)
    print("[*] Applied Taubin smoothing")

    # Subdivide twice for ~30 MB output
    sub = tm.subdivide()
    print(f"[*] After subdivide 1: {len(sub.vertices)} vertices, {len(sub.faces)} faces")
    sub = sub.subdivide()
    print(f"[*] After subdivide 2: {len(sub.vertices)} vertices, {len(sub.faces)} faces")

    # Light smoothing after subdivision
    trimesh.smoothing.filter_taubin(sub, lamb=0.3, nu=0.5, iterations=3)

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
