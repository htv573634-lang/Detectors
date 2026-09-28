"""Compatibility shim: routes torchmcubes calls to PyMCubes + scipy."""
import numpy as np

try:
    import mcubes as _mcubes
except ImportError:
    _mcubes = None

try:
    from scipy.ndimage import map_coordinates
except ImportError:
    map_coordinates = None

try:
    import torch
except ImportError:
    torch = None


def marching_cubes(volume, thresh):
    """Drop-in replacement for torchmcubes.marching_cubes."""
    if _mcubes is None:
        raise RuntimeError("PyMCubes not installed; cannot run marching_cubes")
    # Convert tensor to numpy if needed
    if hasattr(volume, "detach"):
        volume = volume.detach().cpu().numpy()
    if hasattr(volume, "numpy"):
        volume = volume.numpy()
    volume = np.asarray(volume, dtype=np.float32)
    verts, faces = _mcubes.marching_cubes(volume, thresh)
    return verts.astype(np.float32), faces.astype(np.int64)


def grid_interp(volume, points):
    """Simplified grid_interp using scipy."""
    if map_coordinates is None:
        raise RuntimeError("scipy not installed; cannot run grid_interp")
    if hasattr(volume, "detach"):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume, dtype=np.float32)
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    shape = np.array(volume.shape, dtype=np.float32)
    coords = (points + 1.0) / 2.0 * (shape - 1.0)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    coords = coords.T
    values = map_coordinates(volume, coords, order=1, mode="nearest")
    return values.astype(np.float32)
