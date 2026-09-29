import os
import struct

TRI = "SAM3DBody-cpp/onnx/body_mesh.tri"
NUM_VERTS = 18439  # known MHR vertex count
NUM_FACES = 73764  # known MHR face count

if not os.path.isfile(TRI):
    print(f"[ERROR] {TRI} not found")
    raise SystemExit(1)

with open(TRI, "rb") as f:
    data = f.read()

size = len(data)
print(f"[*] File size: {size} bytes")
print(f"[*] Expected face data size: {NUM_FACES * 12} bytes")
print(f"[*] Expected + 0 byte header")
print(f"[*] Expected + 4 byte header -> {4 + NUM_FACES * 12}")
print(f"[*] Expected + 8 byte header -> {8 + NUM_FACES * 12}")
print(f"[*] Expected + 12 byte header -> {12 + NUM_FACES * 12}")
print(f"[*] Expected + 16 byte header -> {16 + NUM_FACES * 12}")
print(f"[*] Expected + 20 byte header -> {20 + NUM_FACES * 12}")
print(f"[*] Expected + 24 byte header -> {24 + NUM_FACES * 12}")
print()

# Try every offset from 0 to 128 in 4-byte steps
print("[*] Scanning offsets for valid int32 face data...")
for offset in range(0, 128, 4):
    remaining = size - offset
    if remaining % 12 != 0:
        continue
    face_count = remaining // 12
    if abs(face_count - NUM_FACES) > 100:
        continue

    # Check first 50 faces
    valid = True
    for i in range(50):
        a, b, c = struct.unpack_from("<iii", data, offset + i * 12)
        if not (0 <= a < NUM_VERTS and 0 <= b < NUM_VERTS and 0 <= c < NUM_VERTS):
            valid = False
            break

    if valid:
        print(f"  [✓] Offset {offset:3d}  ->  {face_count} faces  (VALID)")

print()
print("[*] Also trying uint32 interpretation (same as int32 for small values)")

# Try every offset from 0 to 128 in 2-byte steps for uint16
print()
print("[*] Scanning offsets for valid uint16 face data...")
for offset in range(0, 128, 2):
    remaining = size - offset
    if remaining % 6 != 0:
        continue
    face_count = remaining // 6
    if abs(face_count - NUM_FACES) > 100:
        continue

    valid = True
    for i in range(50):
        a, b, c = struct.unpack_from("<HHH", data, offset + i * 6)
        if not (0 <= a < NUM_VERTS and 0 <= b < NUM_VERTS and 0 <= c < NUM_VERTS):
            valid = False
            break

    if valid:
        print(f"  [✓] Offset {offset:3d}  ->  {face_count} faces  (VALID)")

print()
print("[*] If nothing is marked VALID, the format is not simple int/uint indices.")
print("[*] Dumping first 128 bytes for manual inspection:")
with open(TRI, "rb") as f:
    head = f.read(128)
for i in range(0, 128, 16):
    chunk = head[i:i+16]
    hex_str = " ".join(f"{b:02x}" for b in chunk)
    print(f"    {i:4d}: {hex_str}")
