import os

ROOT = "SAM3DBody-cpp"

if not os.path.isdir(ROOT):
    print(f"[ERROR] {ROOT} not found. Did the clone step succeed?")
    raise SystemExit(1)

print(f"[*] Scanning {ROOT}/ for C++ source files...\n")

hits = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    if "/build/" in dirpath or "/onnx/" in dirpath or "/.git/" in dirpath:
        continue
    for fn in filenames:
        if fn.endswith((".cpp", ".cc", ".cxx", ".h", ".hpp")):
            full = os.path.join(dirpath, fn)
            size = os.path.getsize(full)
            hits.append((full, size))

for path, size in sorted(hits):
    print(f"  {size:>10} bytes   {path}")

print(f"\n[*] Total: {len(hits)} C++/header files")

print("\n[*] Files containing key tokens:\n")
for path, _ in hits:
    try:
        with open(path, "r", errors="ignore") as f:
            content = f.read()
    except Exception:
        continue
    tokens_found = []
    for tok in ["LBS", "num_vertices", "skel=", "verts[", "fast_sam_3dbody", "mhr_lbs"]:
        if tok in content:
            tokens_found.append(tok)
    if tokens_found:
        print(f"  {path}  ->  {tokens_found}")

print("\n[*] Done.")
