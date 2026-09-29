import os
import struct

TRI = "SAM3DBody-cpp/onnx/body_mesh.tri"

if not os.path.isfile(TRI):
    print(f"[ERROR] {TRI} not found")
    raise SystemExit(1)

size = os.path.getsize(TRI)
print(f"[*] File size: {size} bytes")
print(f"[*] Size / 12 (int32 triplets): {size / 12:.1f} faces")
print(f"[*] Size / 6 (uint16 triplets): {size / 6:.1f} faces")
print(f"[*] Size / 9 (ASCII 'a b c\\n'):  {size / 9:.1f} faces")

with open(TRI, "rb") as f:
    head = f.read(64)

print(f"\n[*] First 64 bytes (hex): {head.hex()}")
print(f"[*] First 64 bytes (raw): {head!r}")

# Try interpretations
print("\n[*] Interpreting as int32 triplets (first 5):")
for i in range(5):
    a, b, c = struct.unpack_from("<iii", head, i * 12)
    print(f"    face {i}: {a} {b} {c}")

print("\n[*] Interpreting as uint16 triplets (first 10):")
for i in range(10):
    a, b, c = struct.unpack_from("<HHH", head, i * 6)
    print(f"    face {i}: {a} {b} {c}")

print("\n[*] Interpreting as ASCII text (first line):")
try:
    line = head.split(b"\n")[0]
    print(f"    {line!r}")
except Exception as e:
    print(f"    Failed: {e}")
