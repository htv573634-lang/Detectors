import os
import numpy as np
import trimesh

INPUT_OBJ = "out_sam/test-2_mesh.obj"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"

def smooth_mesh():
    if not os.path.isfile(INPUT_OBJ):
        print(f"[ERROR] {INPUT_OBJ} not found")
        return

    print(f"[*] Loading {INPUT_OBJ}")
    tm = trimesh.load(INPUT_OBJ, force="mesh")

    # Handle both Trimesh and Scene types
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    print(f"[*] Input mesh: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    # 1. Cleanup
    tm.update_faces(tm.nondegenerate_faces())
    tm.update_faces(tm.unique_faces())
    tm.remove_unreferenced_vertices()
    print(f"[*] After cleanup: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    # 2. Taubin smoothing (needs scipy)
    trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=0.53, iterations=10)
    print("[*] Applied Taubin smoothing")

    # 3. Subdivide twice for HQ output (~30 MB)
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
