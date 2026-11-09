from abc import ABC, abstractmethod

import numpy as np


def sample_depth_at_bbox_center(depth_map, x_px, y_px, mask=None):
    """Return the depth value at the bbox center, falling back to the nearest valid masked pixel."""
    if depth_map is None or np.asarray(depth_map).size == 0:
        return None

    depth_array = np.asarray(depth_map, dtype=np.float32)
    if depth_array.ndim == 3 and depth_array.shape[-1] in (1, 3, 4):
        depth_array = depth_array[..., 0]

    if depth_array.ndim != 2:
        return None

    h, w = depth_array.shape[:2]
    cx = int(round(float(x_px)))
    cy = int(round(float(y_px)))

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None

    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape[:2] != depth_array.shape[:2]:
            mask_array = None
        elif not mask_array[cy, cx]:
            ys, xs = np.where(mask_array)
            if ys.size == 0:
                return None
            idx = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
            cx, cy = int(xs[idx]), int(ys[idx])

    value = float(depth_array[cy, cx])
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def normalize_depth_to_meters(value, z_far=200.0):
    """Convert raw model depth values to actual metric distance from the camera in meters."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return None
    if 0.0 < value <= 1.0:
        return value * float(z_far)
    if value > 1000.0:
        return value / 1000.0
    return value


class ReconstructionWorkflow(ABC):
    """Boundary between image detection and a camera reconstruction source."""

    georeferenced = False

    @abstractmethod
    def project_detection(
        self,
        frame_index: int,
        frame_path: str | None,
        x_px: float,
        y_px: float,
        depth_m: float | None,
        class_name: str,
        detection_id: int | str,
        confidence: float,
    ) -> dict | None:
        """Project one detection to world coordinates if possible."""

    def close(self) -> None:
        """Release workflow resources, if any."""
