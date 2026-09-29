import os
import subprocess

SAM_DIR = "SAM3DBody-cpp"

# Find all header files that mention mhr_lbs
print("[*] Searching for mhr_lbs headers...\n")

for dirpath, dirnames, filenames in os.walk(SAM_DIR):
    if "/build" in dirpath or "/.git" in dirpath:
        continue
    for fn in filenames:
        if not fn.endswith((".h", ".hpp")):
            continue
        path = os.path.join(dirpath, fn)
        try:
            with open(path, "r", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        if "mhr_lbs" in content.lower() or "lbs_data" in content.lower():
            print(f"\n=== {path} ===")
            # Print the struct definition
            lines = content.split("\n")
            in_struct = False
            for i, line in enumerate(lines):
                if "struct" in line and ("lbs" in line.lower() or "mhr" in line.lower()):
                    in_struct = True
                if in_struct:
                    print(f"  {i+1:5d}: {line}")
                    if "};" in line:
                        in_struct = False
                # Also grep for 'face' and 'tri' keywords
                if "face" in line.lower() or "tri" in line.lower() or "index" in line.lower():
                    print(f"  >>> {i+1:5d}: {line}")
