import os
import struct

TRI = "SAM3DBody-cpp/onnx/body_mesh.tri"

with open(TRI, "rb") as f:
    data = f.read()

size = len(data)
print(f"[*] File size: {size} bytes")

MAX_INDEX = 20000   # safe upper bound for MHR vertex indices
MIN_RUN = 500       # require at least 500 consecutive valid faces

def check_run(offset, fmt, fmt_size, num_to_check=1000):
    """Return number of consecutive valid faces starting at offset."""
    if offset + num_to_check * fmt_size > size:
        num_to_check = (size - offset) // fmt_size
    if num_to_check < MIN_RUN:
        return 0
    valid = 0
    for i in range(num_to_check):
        try:
            vals = struct.unpack_from(fmt, data, offset + i * fmt_size)
        except struct.error:
            break
        if all(0 <= v < MAX_INDEX for v in vals):
            valid += 1
        else:
            break
    return valid

print("[*] Full-file scan for valid face data...\n")

hits = []

# Scan every byte position, both int32 and uint16 formats
for offset in range(0, size - 12):
    # int32 triplet format (12 bytes each)
    v = check_run(offset, "<iii", 12, num_to_check=200)
    if v >= MIN_RUN:
        hits.append((offset, "int32", v))

    # uint16 triplet format (6 bytes each)
    if offset % 2 == 0:
        v = check_run(offset, "<HHH", 6, num_to_check=200)
        if v >= MIN_RUN:
            hits.append((offset, "uint16", v))

# De-duplicate overlapping hits (keep the first offset in each cluster)
hits.sort()
deduped = []
for offset, fmt, valid in hits:
    if not deduped or offset - deduped[-1][0] > 100:
        deduped.append((offset, fmt, valid))

print(f"[*] Found {len(deduped)} candidate region(s):\n")
for offset, fmt, valid in deduped:
    fmt_size = 12 if fmt == "int32" else 6
    face_count = (size - offset) // fmt_size
    print(f"  Offset {offset:7d}  format={fmt:6s}  first {valid}+ valid faces  (~{face_count} total)")

    # Show first 3 faces
    faces = []
    for i in range(3):
        faces.append(struct.unpack_from(fmt, data, offset + i * fmt_size))
    print(f"    First 3 faces: {faces}")

    # Show last 3 faces if they parse
    last_faces = []
    for i in range(3):
        pos = offset + (face_count - 3 + i) * fmt_size
        if pos + fmt_size <= size:
            try:
                last_faces.append(struct.unpack_from(fmt, data, pos))
            except struct.error:
                break
    print(f"    Last 3 faces:  {last_faces}")
    print()

if not deduped:
    print("[!] No valid face region found anywhere in the file.")
    print("[*] Dumping ASCII content of first 512 bytes:")
    print(data[:512])
