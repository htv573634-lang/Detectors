import os
import re
import sys

SRC = "SAM3DBody-cpp/src/main.cpp"

if not os.path.isfile(SRC):
    print(f"[ERROR] Source file not found: {SRC}")
    sys.exit(1)

with open(SRC, "r") as f:
    content = f.read()

# Multiple candidate markers in order of preference
markers = [
    'verts=" <<',
    'verts=',
    '"skel="',
    '[FSB] LBS',
]

found_marker = None
found_idx = -1

for m in markers:
    idx = content.find(m)
    if idx != -1:
        found_marker = m
        found_idx = idx
        break

if found_idx == -1:
    print("[ERROR] Could not find insertion point. Dumping search context:")
    for m in markers:
        print(f"  Searched for: {m} -> NOT FOUND")
    # Dump lines containing 'LBS' to help debug
    for i, line in enumerate(content.splitlines()):
        if "LBS" in line or "verts" in line:
            print(f"  Line {i}: {line[:120]}")
    sys.exit(1)

print(f"[OK] Found marker '{found_marker}' at position {found_idx}")

# Find the end of the containing statement (next ';' followed by newline)
semi_idx = content.find(";", found_idx)
if semi_idx == -1:
    print("[ERROR] Could not find end of statement")
    sys.exit(1)

line_end = content.find("\n", semi_idx)
if line_end == -1:
    line_end = semi_idx + 1

insert_pos = line_end + 1

# The C++ injection block
inject_code = """
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

new_content = content[:insert_pos] + inject_code + content[insert_pos:]

# Check if already patched to avoid duplicates
if "AUTO-INJECTED MESH EXPORT" in content:
    print("[SKIP] File already patched. No changes made.")
    sys.exit(0)

with open(SRC, "w") as f:
    f.write(new_content)

print("[OK] Patch applied successfully to main.cpp")
