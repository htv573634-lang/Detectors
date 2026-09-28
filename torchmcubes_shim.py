"""Compatibility shim: routes torchmcubes calls to PyMCubes + scipy."""
import numpy as np
import torch

try:
    import mcubes as _mcubes
except ImportError:
    _mcubes = None

try:
    from scipy.ndimage import map_coordinates
except ImportError:
    map_coordinates = None


def marching_cubes(volume, thresh):
    """Drop-in replacement for torchmcubes.marching_cubes.

    IMPORTANT: returns torch tensors (not numpy) because TripoSR
    calls .to(device) on them.
    """
    if _mcubes is None:
        raise RuntimeError("PyMCubes not installed; cannot run marching_cubes")

    # Convert tensor -> numpy for PyMCubes
    if hasattr(volume, "detach"):
        volume_np = volume.detach().cpu().numpy()
    elif hasattr(volume, "numpy"):
        volume_np = volume.numpy()
    else:
        volume_np = np.asarray(volume)
    volume_np = np.asarray(volume_np, dtype=np.float32)

    # PyMCubes returns float32 verts and int64 faces
    verts, faces = _mcubes.marching_cubes(volume_np, float(thresh))

    # Convert to torch tensors (TripoSR calls .to(device) on these)
    verts_t = torch.from_numpy(np.asarray(verts, dtype=np.float32))
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int64))

    return verts_t, faces_t


def grid_interp(volume, points):
    """Simplified grid_interp using scipy. Returns a torch tensor."""
    if map_coordinates is None:
        raise RuntimeError("scipy not installed; cannot run grid_interp")

    if hasattr(volume, "detach"):
        volume_np = volume.detach().cpu().numpy()
    else:
        volume_np = np.asarray(volume)
    volume_np = np.asarray(volume_np, dtype=np.float32)

    if hasattr(points, "detach"):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = np.asarray(points)
    points_np = np.asarray(points_np, dtype=np.float32)

    shape = np.array(volume_np.shape, dtype=np.float32)
    coords = (points_np + 1.0) / 2.0 * (shape - 1.0)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    coords = coords.T
    values = map_coordinates(volume_np, coords, order=1, mode="nearest")
    return torch.from_numpy(values.astype(np.float32))
