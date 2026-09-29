import os
import numpy as np

RAW_VERTICES = "out_sam/mesh.vertices"
TRI_FILE = "SAM3DBody-cpp/onnx/body_mesh.tri"
OUTPUT_OBJ = "out_sam/test-2_mesh.obj"

VERT_COUNT = 18439
TRI_COUNT = 36874
HEADER_SIZE = 144
INDEX_OFFSET = HEADER_SIZE + (VERT_COUNT * 3 * 4) * 2  # = 442680

# Set this to True if the mesh appears upside down in your viewer
FLIP_UPRIGHT = True

def load_posed_vertices(path):
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
    return np.array(verts, dtype=np.float32)

def load_indices(path):
    with open(path, "rb") as f:
        data = f.read()
    print(f"[*] .tri size: {len(data)} bytes")
    i_bytes = TRI_COUNT * 3 * 4
    indices = np.frombuffer(
        data[INDEX_OFFSET:INDEX_OFFSET + i_bytes], dtype=np.int32
    ).reshape(-1, 3)
    print(f"[*] Parsed {len(indices)} triangles")
    print(f"[*] Index range: {indices.min()} to {indices.max()}")
    if indices.max() >= VERT_COUNT:
        print(f"[!] WARNING: max index >= {VERT_COUNT}")
    return indices

def write_obj(verts, faces, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Generated from mesh.vertices + body_mesh.tri\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[OK] Wrote {out_path}")

def main():
    if not os.path.isfile(RAW_VERTICES):
        print(f"[ERROR] {RAW_VERTICES} not found")
        return
    if not os.path.isfile(TRI_FILE):
        print(f"[ERROR] {TRI_FILE} not found")
        return

    verts = load_posed_vertices(RAW_VERTICES)
    print(f"[*] Loaded {len(verts)} posed vertices")
    print(f"    X: {verts[:,0].min():.3f} to {verts[:,0].max():.3f}")
    print(f"    Y: {verts[:,1].min():.3f} to {verts[:,1].max():.3f}")
    print(f"    Z: {verts[:,2].min():.3f} to {verts[:,2].max():.3f}")

    if len(verts) != VERT_COUNT:
        print(f"[WARN] Expected {VERT_COUNT} vertices, got {len(verts)}")

    # Flip to upright orientation if requested
    if FLIP_UPRIGHT:
        print("[*] Flipping mesh 180 degrees around X-axis to make it upright...")
        verts[:, 1] = -verts[:, 1]
        verts[:, 2] = -verts[:, 2]
        print(f"    New Y: {verts[:,1].min():.3f} to {verts[:,1].max():.3f}")
        print(f"    New Z: {verts[:,2].min():.3f} to {verts[:,2].max():.3f}")

    faces = load_indices(TRI_FILE)
    write_obj(verts, faces, OUTPUT_OBJ)

if __name__ == "__main__":
    main()
