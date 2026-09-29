import os
import trimesh
import open3d as o3d
import numpy as np

INPUT_OBJ = "out_sam/test-2_mesh_refined.obj"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"

def smooth_mesh():
    if not os.path.isfile(INPUT_OBJ):
        print(f"[ERROR] {INPUT_OBJ} not found. Skipping HQ generation.")
        return

    print(f"[*] Loading {INPUT_OBJ}")
    tm = trimesh.load(INPUT_OBJ, force="mesh", process=False)
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    print(f"[*] Input: {len(tm.vertices)} verts, {len(tm.faces)} faces")

    # Convert to Open3D for better smoothing
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
    
    # Clean up geometry
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    
    # Taubin smoothing (preserves volume, fixes spikes)
    print("[*] Applying Open3D Taubin smoothing...")
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=8)
    o3d_mesh.compute_vertex_normals()
    
    tm_smooth = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles)
    )
    
    # Subdivide twice for high poly count
    sub = tm_smooth.subdivide()
    sub = sub.subdivide()
    print(f"[*] Subdivided: {len(sub.vertices)} verts, {len(sub.faces)} faces")

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
