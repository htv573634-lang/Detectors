import os
import numpy as np
import pandas as pd
import trimesh
import open3d as o3d

INPUT_OBJ = "out_sam/test-2_mesh.obj"
KEYPOINTS_CSV = "out_sam/test-2_keypoints.csv"
OUTPUT_OBJ = "out_sam/test-2_mesh_fixed.obj"

def get_joint(row, name):
    return np.array([row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]])

def load_abdomen_region(csv_path):
    """Compute abdomen center and radius from skeleton joints. Works for any pose."""
    df = pd.read_csv(csv_path)
    row = df.iloc[0]

    left_hip = get_joint(row, "left_hip")
    right_hip = get_joint(row, "right_hip")
    left_shoulder = get_joint(row, "left_shoulder")
    right_shoulder = get_joint(row, "right_shoulder")

    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    # Abdomen is the midpoint between hip and shoulder centers
    abdomen_center = (hip_center + shoulder_center) / 2.0

    # Radius scales with torso length (distance hip to shoulder)
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    radius = torso_length * 0.55

    print(f"[*] Hip center:      {hip_center.round(3)}")
    print(f"[*] Shoulder center: {shoulder_center.round(3)}")
    print(f"[*] Abdomen center:  {abdomen_center.round(3)}")
    print(f"[*] Torso length:    {torso_length:.3f} m")
    print(f"[*] Region radius:   {radius:.3f} m")

    return abdomen_center, radius

def fix_abdomen():
    if not os.path.isfile(INPUT_OBJ):
        print(f"[ERROR] {INPUT_OBJ} not found")
        return
    if not os.path.isfile(KEYPOINTS_CSV):
        print(f"[ERROR] {KEYPOINTS_CSV} not found")
        return

    # 1. Load mesh
    tm = trimesh.load(INPUT_OBJ, force="mesh", process=False)
    if hasattr(tm, "geometry"):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    verts = np.asarray(tm.vertices)
    faces = np.asarray(tm.faces)
    print(f"[*] Mesh: {len(verts)} vertices, {len(faces)} faces")

    # 2. Compute abdomen region from skeleton
    abdomen_center, radius = load_abdomen_region(KEYPOINTS_CSV)

    # 3. Find vertices inside the abdomen sphere
    dists = np.linalg.norm(verts - abdomen_center, axis=1)
    abdomen_mask = dists < radius
    print(f"[*] Abdomen vertices: {abdomen_mask.sum()} / {len(verts)}")

    if abdomen_mask.sum() < 20:
        print("[WARN] Too few vertices. Skipping deformation.")
        tm.export(OUTPUT_OBJ)
        return

    # 4. Fixed vertices: everything outside the region (plus a buffer ring)
    buffer_mask = dists < radius * 1.15    # slightly bigger than abdomen
    fixed_ids = np.where(~buffer_mask)[0]

    # 5. Build Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    # 6. Create a smoothed version to use as a target for the abdomen vertices
    smoothed = o3d_mesh.filter_smooth_taubin(number_of_iterations=20)
    smooth_verts = np.asarray(smoothed.vertices)

    # 7. Build ARAP constraints:
    #    - fixed vertices stay in place
    #    - abdomen vertices are pulled toward their smoothed positions
    handle_ids = np.where(abdomen_mask)[0]
    constraint_ids = np.concatenate([fixed_ids, handle_ids]).astype(np.int32)
    constraint_positions = np.vstack([
        verts[fixed_ids],
        smooth_verts[handle_ids]
    ])

    print("[*] Applying ARAP deformation...")
    deformed = o3d_mesh.deform_as_rigid_as_possible(
        o3d.utility.IntVector(constraint_ids),
        o3d.utility.Vector3dVector(constraint_positions),
        max_iter=20,
        energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed
    )

    # 8. Export
    result = trimesh.Trimesh(
        vertices=np.asarray(deformed.vertices),
        faces=np.asarray(deformed.triangles)
    )
    result.export(OUTPUT_OBJ)
    size_mb = os.path.getsize(OUTPUT_OBJ) / (1024 * 1024)
    print(f"[OK] Exported {OUTPUT_OBJ} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    fix_abdomen()
