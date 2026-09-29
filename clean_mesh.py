import os
import numpy as np
import trimesh
import open3d as o3d

INPUT_OBJ = "out_sam/test-2_mesh.obj"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"

def smooth_mesh():
    if not os.path.isfile(INPUT_OBJ):
        print(f"[ERROR] {INPUT_OBJ} not found")
        return

    print(f"[*] Loading {INPUT_OBJ}")
    tm = trimesh.load(INPUT_OBJ, force="mesh", process=False)
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    print(f"[*] Input mesh: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    tm.update_faces(tm.nondegenerate_faces())
    tm.update_faces(tm.unique_faces())
    tm.remove_unreferenced_vertices()
    print(f"[*] After cleanup: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))

    print("[*] Applying Open3D Taubin smoothing...")
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=8)
    o3d_mesh.compute_vertex_normals()

    tm_smooth = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles)
    )
    print(f"[*] After smoothing: {len(tm_smooth.vertices)} vertices, {len(tm_smooth.faces)} faces")

    sub = tm_smooth.subdivide()
    print(f"[*] After subdivide 1: {len(sub.vertices)} vertices, {len(sub.faces)} faces")
    sub = sub.subdivide()
    print(f"[*] After subdivide 2: {len(sub.vertices)} vertices, {len(sub.faces)} faces")

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
