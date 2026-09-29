import os
import numpy as np
import trimesh
import open3d as o3d

INPUT_OBJ = "out_sam/test-2_mesh_refined.obj"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"

def smooth_mesh():
    if not os.path.isfile(INPUT_OBJ):
        print(f"[WARN] {INPUT_OBJ} not found, using test-2_mesh.obj")
        INPUT_OBJ_LOCAL = "out_sam/test-2_mesh.obj"
    else:
        INPUT_OBJ_LOCAL = INPUT_OBJ

    print(f"[*] Loading {INPUT_OBJ_LOCAL}")
    tm = trimesh.load(INPUT_OBJ_LOCAL, force="mesh", process=False)
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    print(f"[*] Input: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    # Light Taubin to remove depth-refinement spikes
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=4)
    o3d_mesh.compute_vertex_normals()

    tm_smooth = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles)
    )

    # 2× subdivision for smooth rendering (no shape change)
    sub = tm_smooth.subdivide()
    sub = sub.subdivide()
    print(f"[*] After 2x subdivision: {len(sub.vertices)} vertices, {len(sub.faces)} faces")

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
