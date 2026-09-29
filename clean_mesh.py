import trimesh
import os

def smooth_mesh(input_obj, output_obj):
    print(f"[*] Loading {input_obj}")
    mesh = trimesh.load(input_obj)

    print(f"[*] Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # 1. Remove degenerate faces (fixes spiky holes)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    print(f"[*] After cleanup: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # 2. Taubin smoothing (removes LBS artifacts without shrinking)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=10)
    print("[*] Taubin smoothing applied")

    # 3. Subdivide twice for high poly count (~30 MB)
    sub_mesh = mesh.subdivide()
    print(f"[*] After subdivide 1: {len(sub_mesh.vertices)} vertices, {len(sub_mesh.faces)} faces")
    sub_mesh = sub_mesh.subdivide()
    print(f"[*] After subdivide 2: {len(sub_mesh.vertices)} vertices, {len(sub_mesh.faces)} faces")

    # 4. Re-apply light smoothing after subdivision
    trimesh.smoothing.filter_taubin(sub_mesh, lamb=0.3, nu=0.5, iterations=3)

    # 5. Ensure output dir exists
    os.makedirs(os.path.dirname(output_obj), exist_ok=True)

    # 6. Export
    sub_mesh.export(output_obj)
    size_mb = os.path.getsize(output_obj) / (1024 * 1024)
    print(f"[OK] Exported {output_obj} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    smooth_mesh(
        "out_sam/test-2_mesh.obj",
        "out_sam/test-2_mesh_hq.obj"
    )
