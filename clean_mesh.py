import os
import numpy as np
import trimesh

RAW_VERTICES = "out_sam/mesh.vertices"
FALLBACK_OBJ = "out_sam/test-2_mesh.obj"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"

def load_raw_vertices(path):
    """Load the space-separated x y z file written by the C++ patch."""
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
    """Reconstruct a watertight mesh from a point cloud using Open3D Poisson."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)

    # Estimate normals (radius is in mesh units; human body ~1.8 tall)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    print("[*] Running Poisson reconstruction (depth=9)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    # Crop to the original point cloud bounding box to remove padding
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)

    # Clean up
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    print(f"[*] Reconstructed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
    )

def smooth_mesh():
    # 1. Prefer raw vertices + Poisson
    if os.path.isfile(RAW_VERTICES):
        print(f"[*] Loading raw vertices from {RAW_VERTICES}")
        verts = load_raw_vertices(RAW_VERTICES)
        print(f"[*] Loaded {len(verts)} vertices")
        try:
            tm = poisson_reconstruct(verts)
        except Exception as e:
            print(f"[WARN] Poisson failed: {e}")
            print("[*] Falling back to loading the existing OBJ")
            tm = trimesh.load(FALLBACK_OBJ)
    else:
        print(f"[WARN] {RAW_VERTICES} not found, loading {FALLBACK_OBJ}")
        tm = trimesh.load(FALLBACK_OBJ, force="mesh")

    print(f"[*] Input mesh: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    # 2. Taubin smoothing to remove LBS artifacts
    trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=0.53, iterations=10)
    print("[*] Applied Taubin smoothing")

    # 3. Subdivide twice for ~30 MB output
    sub = tm.subdivide()
    print(f"[*] After subdivide 1: {len(sub.vertices)} vertices, {len(sub.faces)} faces")
    sub = sub.subdivide()
    print(f"[*] After subdivide 2: {len(sub.vertices)} vertices, {len(sub.faces)} faces")

    # 4. Light smoothing after subdivision
    trimesh.smoothing.filter_taubin(sub, lamb=0.3, nu=0.5, iterations=3)

    # 5. Export
    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
