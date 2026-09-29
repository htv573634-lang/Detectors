import os
import numpy as np
import trimesh

MESH = "out_sam/test-2_mesh.obj"
KEYPOINTS = "out_sam/test-2_keypoints.csv"

IMG_SIZE = 512
CX, CY = 256.0, 256.0
FOCAL = 718.9
CAM_T = np.array([0.017, 0.571, 2.065])

print("="*70)
print("MESH ANALYSIS")
print("="*70)
mesh = trimesh.load(MESH, force="mesh", process=False)
if hasattr(mesh, "geometry"):
    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
verts = mesh.vertices

print(f"Vertices: {len(verts)}")
print(f"X: {verts[:,0].min():.4f} to {verts[:,0].max():.4f} (mean {verts[:,0].mean():.4f})")
print(f"Y: {verts[:,1].min():.4f} to {verts[:,1].max():.4f} (mean {verts[:,1].mean():.4f})")
print(f"Z: {verts[:,2].min():.4f} to {verts[:,2].max():.4f} (mean {verts[:,2].mean():.4f})")

print()
print("="*70)
print("KEYPOINTS ANALYSIS (CSV)")
print("="*70)
if os.path.isfile(KEYPOINTS):
    with open(KEYPOINTS, "r") as f:
        header = f.readline().strip().split(",")
        row = f.readline().strip().split(",")
    
    # Skip frame and skeleton_id, get x,y,z triplets
    kp_vals = [float(v) for v in row[2:]]
    kp = np.array(kp_vals).reshape(-1, 3)
    print(f"Keypoints: {len(kp)}")
    print(f"X: {kp[:,0].min():.4f} to {kp[:,0].max():.4f} (mean {kp[:,0].mean():.4f})")
    print(f"Y: {kp[:,1].min():.4f} to {kp[:,1].max():.4f} (mean {kp[:,1].mean():.4f})")
    print(f"Z: {kp[:,2].min():.4f} to {kp[:,2].max():.4f} (mean {kp[:,2].mean():.4f})")
else:
    print("Keypoints file not found.")

print()
print("="*70)
print("PROJECTION TESTS (looking for v in 0..512, u in 0..512)")
print("="*70)

def test_projection(name, x, y, z):
    # Distance from camera in OpenGL convention (looking down -Z)
    # Try distance = -z, distance = z, distance = z + cam_t, distance = -z + cam_t
    for d_name, dist in [
        ("d=-z",      -z),
        ("d=z",        z),
        ("d=z+cam_t",  z + CAM_T[2]),
        ("d=cam_t-z",  CAM_T[2] - z),
    ]:
        valid = dist > 0.1
        if valid.sum() == 0:
            continue
        u = x / np.where(valid, dist, 1.0) * FOCAL + CX
        v = -y / np.where(valid, dist, 1.0) * FOCAL + CY
        u_ok = (u > 0) & (u < IMG_SIZE)
        v_ok = (v > 0) & (v < IMG_SIZE)
        both = valid & u_ok & v_ok
        if both.sum() == 0:
            continue
        print(f"  [{name}] {d_name}: valid={both.sum():5d}  u=[{u[both].min():7.1f},{u[both].max():7.1f}]  v=[{v[both].min():7.1f},{v[both].max():7.1f}]")

# Test various sign conventions on Y and Z translations
tests = [
    ("raw",         verts[:,0],                      verts[:,1],                     verts[:,2]                    ),
    ("+t",          verts[:,0]+CAM_T[0],             verts[:,1]+CAM_T[1],            verts[:,2]+CAM_T[2]           ),
    ("-t",          verts[:,0]-CAM_T[0],             verts[:,1]-CAM_T[1],            verts[:,2]-CAM_T[2]           ),
    ("y+,z+",       verts[:,0]+CAM_T[0],             verts[:,1]+CAM_T[1],            verts[:,2]+CAM_T[2]           ),
    ("y+,z-",       verts[:,0]+CAM_T[0],             verts[:,1]+CAM_T[1],            verts[:,2]-CAM_T[2]           ),
    ("y-,z+",       verts[:,0]+CAM_T[0],             verts[:,1]-CAM_T[1],            verts[:,2]+CAM_T[2]           ),
    ("y-,z-",       verts[:,0]+CAM_T[0],             verts[:,1]-CAM_T[1],            verts[:,2]-CAM_T[2]           ),
    ("-y,-z",       verts[:,0]+CAM_T[0],            -verts[:,1]-CAM_T[1],           -verts[:,2]-CAM_T[2]          ),
    ("-y,-z+",      verts[:,0]+CAM_T[0],            -verts[:,1]+CAM_T[1],           -verts[:,2]+CAM_T[2]          ),
    ("flipy",       verts[:,0]+CAM_T[0],            -verts[:,1]+CAM_T[1],            verts[:,2]-CAM_T[2]           ),
    ("flipz",       verts[:,0]+CAM_T[0],             verts[:,1]-CAM_T[1],           -verts[:,2]+CAM_T[2]          ),
    ("y_only",      verts[:,0],                      verts[:,1]-CAM_T[1],            verts[:,2]-CAM_T[2]           ),
]

for name, x, y, z in tests:
    test_projection(name, x, y, z)

print()
print("="*70)
print("INTERPRETATION")
print("="*70)
print("We want a row where valid is close to 18439 and u/v are within 0-512.")
print("If multiple rows work, we pick the one with v range closest to 50-460.")
