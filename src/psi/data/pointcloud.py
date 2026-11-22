"""Optional point-cloud decoding shared by the raw-data converters."""

import re
from pathlib import Path

import numpy as np


def load_lidar_points(path: Path) -> np.ndarray | None:
    """Decode a timestamp-named PCD, retaining nonzero XYZ rows as float32.

    Missing/empty point clouds are optional and return None. Missing dependencies,
    decoder failures and invalid arrays are errors, not absent observations.
    """

    def pad_timestamp(match: re.Match) -> str:
        return f"{match['whole']}.{match['dec'].ljust(6, '0')}"

    normalized = Path(
        re.sub(r"(?P<whole>\d+)\.(?P<dec>\d{1,6})(?=\.pcd$)", pad_timestamp, str(path))
    )
    if not normalized.is_file():
        return None
    try:
        import open3d
    except ModuleNotFoundError as exc:
        if exc.name == "open3d":
            raise ModuleNotFoundError(
                "Reading PCD observations requires the optional open3d dependency; "
                "install it in the conversion environment before processing lidar."
            ) from exc
        raise
    points = np.asarray(
        open3d.io.read_point_cloud(str(normalized)).points, dtype=np.float32
    )
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"Invalid XYZ point cloud: {normalized}")
    points = points[~np.all(points == 0, axis=1)]
    return points if points.size else None
