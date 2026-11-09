import numpy as np

from reconstruction.base import sample_depth_at_bbox_center


def test_sample_depth_at_bbox_center_uses_bbox_center():
    depth = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 9.5, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    mask = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=bool)

    assert sample_depth_at_bbox_center(depth, 1.0, 1.0, mask) == 9.5
