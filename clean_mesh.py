import os
import trimesh

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

    trimesh.smoothing.filter_taubin(tm, lamb=0.4, nu=0.5, iterations=5)
    print("[*] Applied Taubin smoothing")

    sub = tm.subdivide()
    print(f"[*] After subdivide: {len(sub.vertices)} vertices, {len(sub.faces)} faces")

    os.makedirs(os.path.dirname(OUTPUT_OBJ), exist_ok=True)
    sub.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh()
