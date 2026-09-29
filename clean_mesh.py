import os
import trimesh

INPUT_OBJ = "out_sam/test-2_mesh_refined.obj"
OUTPUT_OBJ = "out_sam/test-2_mesh_hq.obj"
NORMAL_MAP = "out_sam/test-2_normal_map.png"

def smooth_mesh():
    if not os.path.isfile(INPUT_OBJ):
        print(f"[ERROR] {INPUT_OBJ} not found")
        return

    print(f"[*] Loading {INPUT_OBJ}")
    tm = trimesh.load(INPUT_OBJ, force="mesh", process=False)
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    print(f"[*] Input: {len(tm.vertices)} verts, {len(tm.faces)} faces")

    # Light smoothing only — preserve curvature
    trimesh.smoothing.filter_taubin(tm, lamb=0.2, nu=0.5, iterations=3)

    # TWO subdivisions for smooth curves (Catmull-Clark-like)
    sub = tm.subdivide()
    sub = sub.subdivide()
    print(f"[*] Subdivided: {len(sub.vertices)} verts, {len(sub.faces)} faces")

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

    # Report normal map availability for texture step
    if os.path.isfile(NORMAL_MAP):
        print(f"[*] Normal map available for use as texture: {NORMAL_MAP}")

if __name__ == "__main__":
    smooth_mesh()
