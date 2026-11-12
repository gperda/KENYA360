import csv
import sys
import types
from types import SimpleNamespace

import numpy as np

from reconstruction.colmap import ColmapWorkflow
from scripts.colmap_pipeline import write_trajectory


def test_colmap_workflow_projects_trajectory_detection(tmp_path):
    csv_path = tmp_path / "trajectory.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "qw", "qx", "qy", "qz"])
        writer.writerow(["frame_000", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    config = SimpleNamespace(
        trajectory_file_path=str(csv_path),
        input_dir=str(tmp_path),
    )

    workflow = ColmapWorkflow(config)
    result = workflow.project_detection(
        frame_index=0,
        frame_path="/tmp/frame_000.jpg",
        x_px=960.0,
        y_px=540.0,
        depth_m=5.0,
        class_name="buffalo",
        detection_id=7,
        confidence=0.9,
    )

    assert result is not None
    assert result["Image"] == "frame_000"
    assert result["X"] == 0.0
    assert result["Y"] == 0.0
    assert result["Z"] == 5.0
    assert result["distance_from_camera"] == 5.0
    assert result["type"] == "buffalo"
    assert result["id"] == 7
    assert result["confidence"] == 0.9


def test_colmap_workflow_missing_depth_returns_none(tmp_path):
    csv_path = tmp_path / "trajectory.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "qw", "qx", "qy", "qz"])
        writer.writerow(["frame_000", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    workflow = ColmapWorkflow(SimpleNamespace(trajectory_file_path=str(csv_path), input_dir=str(tmp_path)))
    assert workflow.project_detection(
        frame_index=0,
        frame_path="frame_000.jpg",
        x_px=960.0,
        y_px=540.0,
        depth_m=None,
        class_name="buffalo",
        detection_id=1,
        confidence=0.5,
    ) is None


def test_colmap_workflow_reads_optimized_txt_quaternion_format(tmp_path):
    txt_path = tmp_path / "trajectory_optimized.txt"
    txt_path.write_text(
        "# image_name tx ty tz qw qx qy qz\n"
        "# camera-to-world pose: (tx, ty, tz) camera center, (qw, qx, qy, qz) camera orientation in world\n"
        "frame_000.jpg 0.0 1.0 2.0 1.0 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    workflow = ColmapWorkflow(SimpleNamespace(trajectory_file_path=str(txt_path), input_dir=str(tmp_path)))
    pose = workflow._load_trajectory(str(txt_path))["frame_000"]

    assert pose[0].tolist() == [0.0, 1.0, 2.0]
    assert np.allclose(pose[1], np.eye(3))


def test_write_trajectory_projected_to_epsg_cartesian(tmp_path, monkeypatch):
    class FakeTransformer:
        @classmethod
        def from_crs(cls, *_args, **_kwargs):
            return cls()

        def transform(self, x, y, z):
            return (x + 10.0, y + 20.0, z + 30.0)

    monkeypatch.setitem(sys.modules, "pyproj", types.SimpleNamespace(Transformer=FakeTransformer))

    traj = [("frame_000", np.eye(3), np.array([1.0, 2.0, 3.0]))]
    out_path = tmp_path / "projected.csv"
    residuals = write_trajectory(
        out_path,
        traj,
        {"frame_000": np.array([5.0, 6.0, 7.0])},
        "cartesian",
        "EPSG:32637",
        None,
    )

    assert residuals == [np.linalg.norm(np.array([11.0, 22.0, 33.0]) - np.array([15.0, 26.0, 37.0]))]

    with out_path.open(newline="") as fh:
        rows = list(csv.reader(fh))

    assert rows[0][0:8] == ["frame", "x", "y", "z", "qw", "qx", "qy", "qz"]
    assert rows[0][8:20] == ["omega", "phi", "kappa", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]
    assert rows[1][1:4] == ["11.0000", "22.0000", "33.0000"]
    assert len(rows[0]) == 21
