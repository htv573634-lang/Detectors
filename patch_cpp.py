import os
import sys

ROOT = "SAM3DBody-cpp"

def find_candidates():
    candidates = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if "/build" in dirpath or "/onnx" in dirpath or "/.git" in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith((".cpp", ".cc", ".cxx")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            score = 0
            if "num_vertices" in content: score += 3
            if "verts[" in content:       score += 3
            if "skel=" in content:        score += 2
            if "LBS" in content:          score += 2
            if "fast_sam_3dbody" in content: score += 1
            if "mhr_lbs" in content:      score += 1

            if score >= 4:
                candidates.append((path, score, content))
    return sorted(candidates, key=lambda x: -x[1])

def patch_file(path, content):
    markers = ['"skel="', 'skel=', 'num_vertices']

    insert_idx = -1
    used_marker = None
    for m in markers:
        idx = content.find(m)
        if idx != -1:
            insert_idx = idx
            used_marker = m
            break

    if insert_idx == -1:
        return None, "no marker found"

    semi = content.find(";", insert_idx)
    if semi == -1:
        return None, "no semicolon after marker"
    nl = content.find("\n", semi)
    if nl == -1:
        nl = semi + 1
    insert_pos = nl + 1

    inject = """
        // === AUTO-INJECTED MESH EXPORT (patch_cpp.py) ===
        {
            std::ofstream obj_file("out_sam/mesh.vertices");
            for (int pi = 0; pi < num_vertices; ++pi) {
                obj_file << verts[pi*3] << " "
                         << verts[pi*3+1] << " "
                         << verts[pi*3+2] << "\\n";
            }
            obj_file.close();
            std::cout << "[PATCH] Exported " << num_vertices
                      << " vertices to out_sam/mesh.vertices" << std::endl;
        }
        // === END AUTO-INJECTED MESH EXPORT ===
"""

    if "AUTO-INJECTED MESH EXPORT" in content:
        return None, "already patched"

    new_content = content[:insert_pos] + inject + content[insert_pos:]
    with open(path, "w") as f:
        f.write(new_content)
    return used_marker, None

def main():
    if not os.path.isdir(ROOT):
        print(f"[ERROR] {ROOT} not found.")
        sys.exit(1)

    candidates = find_candidates()
    if not candidates:
        print("[ERROR] No suitable .cpp file found containing LBS/verts tokens.")
        print("        Listing all .cpp files for inspection:")
        for dirpath, dirnames, filenames in os.walk(ROOT):
            if "/build" in dirpath or "/.git" in dirpath:
                continue
            for fn in filenames:
                if fn.endswith((".cpp", ".cc", ".cxx")):
                    print(f"        {os.path.join(dirpath, fn)}")
        sys.exit(1)

    print(f"[*] Found {len(candidates)} candidate(s):")
    for path, score, _ in candidates:
        print(f"    {path}  (score={score})")

    for path, score, content in candidates:
        marker, err = patch_file(path, content)
        if err:
            print(f"[SKIP] {path}: {err}")
            continue
        print(f"[OK] Patched {path} (marker='{marker}')")
        os.makedirs(os.path.join(ROOT, "out_sam"), exist_ok=True)
        return

    print("[ERROR] All candidates failed to patch.")
    sys.exit(1)

if __name__ == "__main__":
    main()
