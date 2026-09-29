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

    # LOWER depth = less webbing between close body parts
    print("[*] Running Poisson reconstruction (depth=7)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=7
    )

    # Aggressive crop to remove padding
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.0, bbox.get_center())  # NO expansion — cut tighter
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

def remove_webbing(tm):
    """Delete faces with extremely long edges (the bridge artifacts)."""
    print("[*] Removing webbing (long-edge faces)...")
    edge_lengths = tm.edges_unique_length
    threshold = np.percentile(edge_lengths, 95) * 1.5

    bad = []
    for i, face in enumerate(tm.faces):
        v0, v1, v2 = tm.vertices[face]
        d1 = np.linalg.norm(v0 - v1)
        d2 = np.linalg.norm(v1 - v2)
        d3 = np.linalg.norm(v2 - v0)
        if max(d1, d2, d3) > threshold:
            bad.append(i)

    if bad:
        keep = np.setdiff1d(np.arange(len(tm.faces)), bad)
        tm.update_faces(keep)
        tm.remove_unreferenced_vertices()
        print(f"[*] Removed {len(bad)} webbing faces")
    else:
        print("[*] No webbing detected")

def smooth_mesh():
    if not os.path.isfile(RAW_VERTICES):
        print(f"[ERROR] {RAW_VERTICES} not found")
        return

    verts = load_raw_vertices(RAW_VERTICES)
    print(f"[*] Loaded {len(verts)} vertices")

    tm = poisson_reconstruct(verts)
    remove_webbing(tm)

    # Lighter smoothing (too much smoothing makes the mesh plastic-looking)
    trimesh.smoothing.filter_taubin(tm, lamb=0.4, nu=0.5, iterations=5)
    print("[*] Applied Taubin smoothing (lighter)")

    # SINGLE subdivision instead of two (webbing becomes more visible with each pass)
    sub = tm.subdivide()
    print(f"[*] After subdivide: {len(sub.vertices)} vertices, {len(sub.faces)} faces")

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
