import os
import trimesh

INPUT_OBJ = "out_sam/test-2_mesh_refined.obj"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"
NORMAL_MAP = "out_sam/test-2_depth_vis.png"

def smooth_mesh():
    if not os.path.isfile(INPUT_OBJ):
        print(f"[ERROR] {INPUT_OBJ} not found")
        return

    print(f"[*] Loading {INPUT_OBJ}")
    tm = trimesh.load(INPUT_OBJ, force="mesh", process=False)
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    print(f"[*] Input: {len(tm.vertices)} verts, {len(tm.faces)} faces")

    # VERY light smoothing (1 iteration) to preserve depth curves
    trimesh.smoothing.filter_taubin(tm, lamb=0.1, nu=0.5, iterations=1)

    # Subdivide twice for smooth rendering
    sub = tm.subdivide()
    sub = sub.subdivide()
    print(f"[*] Subdivided: {len(sub.vertices)} verts, {len(sub.faces)} faces")

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
