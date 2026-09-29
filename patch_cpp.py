import os
import sys
import re

TARGET = "SAM3DBody-cpp/src/core/fast_sam_3dbody.cpp"

if not os.path.isfile(TARGET):
    print(f"[ERROR] {TARGET} not found")
    sys.exit(1)

with open(TARGET, "r") as f:
    content = f.read()

# 1. Revert any previously injected block
if "AUTO-INJECTED MESH EXPORT" in content:
    start_marker = "// === AUTO-INJECTED MESH EXPORT"
    end_marker = "// === END AUTO-INJECTED MESH EXPORT ==="
    s = content.find(start_marker)
    e = content.find(end_marker)
    if s != -1 and e != -1:
        e_end = content.find("\n", e) + 1
        content = content[:s] + content[e_end:]
        print("[OK] Reverted previous bad patch.")

# 2. Ensure <fstream> is included
if "#include <fstream>" not in content:
    include_anchor = content.find("#include")
    if include_anchor != -1:
        line_end = content.find("\n", include_anchor) + 1
        content = content[:line_end] + "#include <fstream>\n" + content[line_end:]
        print("[OK] Added #include <fstream>")

# 3. Find the LBS printf statement
pattern = re.compile(r'printf\("\[FSB\] LBS:[^\n]*\);')
match = pattern.search(content)
if not match:
    print("[ERROR] Could not find LBS printf statement")
    sys.exit(1)

insert_pos = match.end()
print(f"[*] Found LBS printf ending at char {insert_pos}")

# 4. Insert the mesh export block
inject = """

        // === AUTO-INJECTED MESH EXPORT ===
        {
            std::ofstream obj_file("out_sam/mesh.vertices");
            size_t n_export = (size_t)meta.num_vertices;
            if (all_verts.size() < n_export * 3) n_export = all_verts.size() / 3;
            for (size_t pi = 0; pi < n_export; ++pi) {
                obj_file << all_verts[pi*3]     << " "
                         << all_verts[pi*3 + 1] << " "
                         << all_verts[pi*3 + 2] << "\\n";
            }
            obj_file.close();
            printf("[PATCH] Exported %zu vertices to out_sam/mesh.vertices\\n", n_export);
        }
        // === END AUTO-INJECTED MESH EXPORT ===
"""

content = content[:insert_pos] + inject + content[insert_pos:]

with open(TARGET, "w") as f:
    f.write(content)

print("[OK] Patch applied successfully to fast_sam_3dbody.cpp")
