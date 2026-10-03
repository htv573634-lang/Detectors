"""
Fuse HMR2.0 (complete 360° body) with MoGe (detailed front surface).
Reads from out_hmr2/ and out_moge/, writes fused mesh to out_hmr2/.
"""
import os
import glob
import json
import time
import numpy as np
import trimesh
import open3d as o3d

# ── CONFIG ──
HMR2_DIR = "out_hmr2"
MOGE_DIR = "out_moge"
OUTPUT_DIR = "out_hmr2"

# Fusion tuning
SEARCH_RADIUS = 0.10          # 10 cm — max distance to search for MoGe point
MAX_DISPLACEMENT = 0.05       # 5 cm — max a vertex can be displaced
FRONT_FACING_THRESHOLD = 0.3  # cos(angle) — dot with camera forward
SMOOTH_ITERS = 5              # Taubin smoothing iterations


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def load_points(path):
    """Load a PLY point cloud, return (N, 3) array."""
    obj = trimesh.load(path, process=False)
    if hasattr(obj, "geometry"):
        verts = []
        for g in obj.geometry.values():
            if hasattr(g, "vertices"):
                verts.append(np.asarray(g.vertices))
        if not verts:
            return None
        return np.vstack(verts)
    if hasattr(obj, "vertices"):
        return np.asarray(obj.vertices)
    return None


def fuse_one(hmr2_path, moge_path, out_dir, base_name):
    log(f"[FUSE] {base_name}")

    # ── Load HMR2.0 base mesh ──
    log(f"  HMR2.0: {hmr2_path}")
    hmr2_mesh = load_mesh(hmr2_path)
    hmr2_verts = np.asarray(hmr2_mesh.vertices).copy()
    hmr2_faces = np.asarray(hmr2_mesh.faces).copy()
    log(f"    {len(hmr2_verts)} verts, {len(hmr2_faces)} faces")

    # ── Load MoGe point cloud ──
    log(f"  MoGe:   {moge_path}")
    moge_pts = load_points(moge_path)
    if moge_pts is None or len(moge_pts) < 100:
        log(f"    [ERROR] MoGe has too few points, skipping")
        return None
    log(f"    {len(moge_pts)} points")

    # ── Align MoGe to HMR2.0 coordinate frame ──
    hmr_min, hmr_max = hmr2_verts.min(0), hmr2_verts.max(0)
    hmr_extent = hmr_max - hmr_min

    moge_min, moge_max = moge_pts.min(0), moge_pts.max(0)
    moge_extent = moge_max - moge_min

    log(f"  Aligning...")
    log(f"    HMR extent:  {hmr_extent.round(3)}")
    log(f"    MoGe extent: {moge_extent.round(3)}")

    # Scale MoGe by height (Y-axis)
    if moge_extent[1] > 1e-6:
        scale = hmr_extent[1] / moge_extent[1]
    else:
        scale = 1.0
    moge_pts_scaled = moge_pts * scale
    log(f"    Scale factor: {scale:.4f}")

    # Recenter MoGe to HMR2.0's center
    hmr_center = (hmr_min + hmr_max) / 2
    moge_center = (moge_pts_scaled.min(0) + moge_pts_scaled.max(0)) / 2
    moge_pts_aligned = moge_pts_scaled + (hmr_center - moge_center)

    # ── Auto-detect Y-axis flip ──
    def count_matches(pts_test, hmr_verts_test, radius):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_test)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        idx = np.random.choice(len(hmr_verts_test), min(500, len(hmr_verts_test)), replace=False)
        count = 0
        for i in idx:
            k, _, d2 = kdtree.search_knn_vector_3d(hmr_verts_test[i], 1)
            if k > 0 and np.sqrt(d2[0]) < radius:
                count += 1
        return count

    option_a = moge_pts_aligned.copy()
    option_b = moge_pts_aligned.copy()
    option_b[:, 1] = -option_b[:, 1]

    match_a = count_matches(option_a, hmr2_verts, SEARCH_RADIUS)
    match_b = count_matches(option_b, hmr2_verts, SEARCH_RADIUS)
    log(f"    Match test: normal={match_a} flipped={match_b}")

    if match_b > match_a:
        moge_pts_aligned = option_b
        log(f"    Flipped MoGe Y-axis to match HMR2.0")
    else:
        moge_pts_aligned = option_a

    # ── Compute HMR2.0 normals ──
    hmr2_mesh.fix_normals()
    hmr_normals = np.asarray(hmr2_mesh.vertex_normals)

    # ── Build KD-tree for MoGe ──
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(moge_pts_aligned)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # ── Displace front-facing vertices ──
    log(f"  Displacing front-facing vertices toward MoGe...")
    log(f"    Search radius: {SEARCH_RADIUS} m")
    log(f"    Max displacement: {MAX_DISPLACEMENT} m")

    refined = hmr2_verts.copy()
    displaced = 0
    skipped_back = 0
    skipped_no_match = 0

    for i in range(len(hmr2_verts)):
        # Only displace front-facing vertices (normal.Z < -threshold)
        if hmr_normals[i, 2] > -FRONT_FACING_THRESHOLD:
            skipped_back += 1
            continue

        k, idx, d2 = kdtree.search_knn_vector_3d(hmr2_verts[i], 1)
        if k == 0:
            skipped_no_match += 1
            continue

        dist = np.sqrt(d2[0])
        if dist > SEARCH_RADIUS:
            skipped_no_match += 1
            continue

        delta = moge_pts_aligned[idx[0]] - hmr2_verts[i]
        delta_len = np.linalg.norm(delta)
        if delta_len < 1e-6:
            continue

        if delta_len > MAX_DISPLACEMENT:
            delta = delta / delta_len * MAX_DISPLACEMENT

        refined[i] = hmr2_verts[i] + delta
        displaced += 1

    log(f"    Displaced: {displaced} vertices")
    log(f"    Skipped (back-facing): {skipped_back}")
    log(f"    Skipped (no MoGe match): {skipped_no_match}")

    # ── Build fused mesh ──
    fused = trimesh.Trimesh(vertices=refined, faces=hmr2_faces, process=False)

    # ── Smooth ──
    log(f"  Taubin smoothing ({SMOOTH_ITERS} iters)...")
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(fused.vertices),
        o3d.utility.Vector3iVector(fused.faces),
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERS)
    o3d_mesh.compute_vertex_normals()

    final = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )

    # ── Export ──
    out_obj = os.path.join(out_dir, f"{base_name}_hmr2_moge_fused.obj")
    out_glb = os.path.join(out_dir, f"{base_name}_hmr2_moge_fused.glb")
    final.export(out_obj)
    final.export(out_glb)
    log(f"  [OK] {out_glb}")
    log(f"       {len(final.vertices)} verts, {len(final.faces)} faces")

    return {
        "name": base_name,
        "hmr2_source": os.path.basename(hmr2_path),
        "moge_source": os.path.basename(moge_path),
        "vertices": int(len(final.vertices)),
        "faces": int(len(final.faces)),
        "displaced_vertices": int(displaced),
        "preserved_back_vertices": int(skipped_back),
        "glb": out_glb,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find HMR2.0 meshes in out_hmr2/
    hmr2_files = glob.glob(os.path.join(HMR2_DIR, "hmr2_*_mesh_*.obj"))
    if not hmr2_files:
        hmr2_files = glob.glob(os.path.join(HMR2_DIR, "hmr2_*_mesh_*.glb"))
    if not hmr2_files:
        log(f"[ERROR] No HMR2.0 meshes in {HMR2_DIR}/")
        return

    log(f"Found {len(hmr2_files)} HMR2.0 mesh(es)")

    index = {}
    for hmr2_path in hmr2_files:
        stem = os.path.splitext(os.path.basename(hmr2_path))[0]
        # Format: hmr2_female_mesh_<name> → <name>
        if "_mesh_" not in stem:
            continue
        base_name = stem.split("_mesh_")[1]

        # Find matching MoGe PLY
        moge_candidates = [
            os.path.join(MOGE_DIR, f"{base_name}_points.ply"),
            os.path.join(MOGE_DIR, base_name, f"{base_name}_points.ply"),
        ]
        moge_path = None
        for c in moge_candidates:
            if os.path.isfile(c):
                moge_path = c
                break

        if moge_path is None:
            # Fallback: any PLY
            fallback = glob.glob(os.path.join(MOGE_DIR, "*_points.ply"))
            if fallback:
                moge_path = fallback[0]

        if moge_path is None:
            log(f"[SKIP] No MoGe PLY for {base_name}")
            continue

        try:
            result = fuse_one(hmr2_path, moge_path, OUTPUT_DIR, base_name)
            if result:
                index[base_name] = result
        except Exception as e:
            log(f"[ERROR] Fusion failed for {base_name}: {e}")
            import traceback
            traceback.print_exc()

    with open(os.path.join(OUTPUT_DIR, "fusion_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    log(f"\n[SUCCESS] Fused {len(index)} character(s)")


if __name__ == "__main__":
    main()
