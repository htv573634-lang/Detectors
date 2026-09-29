import os
import sys
import re

ROOT = "SAM3DBody-cpp"
TARGET = os.path.join(ROOT, "src", "core", "fast_sam_3dbody.cpp")

if not os.path.isfile(TARGET):
    print(f"[ERROR] {TARGET} not found")
    sys.exit(1)

with open(TARGET, "r") as f:
    content = f.read()

# 1. Revert any previous AUTO-INJECTED block
if "AUTO-INJECTED MESH EXPORT" in content:
    start_marker = "// === AUTO-INJECTED MESH EXPORT"
    end_marker = "// === END AUTO-INJECTED MESH EXPORT ==="
    s = content.find(start_marker)
    e = content.find(end_marker)
    if s != -1 and e != -1:
        e_end = content.find("\n", e) + 1
        content = content[:s] + content[e_end:]
        with open(TARGET, "w") as f:
            f.write(content)
        print("[OK] Reverted previous bad patch.")
    else:
        print("[WARN] Markers incomplete; skipping revert.")
else:
    print("[*] No previous patch found.")

# 2. Dump context around the LBS log line
lines = content.split("\n")
found = False
for i, line in enumerate(lines):
    if "LBS" in line and ("cout" in line or "printf" in line or "LOG" in line or "<<" in line):
        print(f"\n=== LBS log statement at line {i+1} ===")
        start = max(0, i - 25)
        end = min(len(lines), i + 45)
        for j in range(start, end):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1:5d}: {lines[j]}")
        found = True
        break

if not found:
    print("[WARN] No line with both 'LBS' and a print statement found.")
    print("Dumping 60 lines after every 'LBS' mention:\n")
    for i, line in enumerate(lines):
        if "LBS" in line:
            print(f"--- LBS at line {i+1} ---")
            for j in range(i, min(len(lines), i + 15)):
                print(f"  {j+1:5d}: {lines[j]}")
            print()

# 3. Also grep for candidate variable declarations
print("\n=== grep: verts / vertices / skel / num_ ===")
for i, line in enumerate(lines):
    if re.search(r'\b(num_vertices|n_verts|num_verts|vertices|verts|skel|n_skel|num_skel)\b', line):
        # Only show lines with assignments or declarations
        if any(tok in line for tok in ["=", "int", "size_t", "auto", "const"]):
            print(f"  {i+1:5d}: {line.strip()[:120]}")

print("\n[*] Diagnostic complete. Paste the output above.")
sys.exit(0)
