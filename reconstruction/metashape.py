import os

import Metashape
import numpy as np

from reconstruction.base import ReconstructionWorkflow


class MetashapeWorkflow(ReconstructionWorkflow):
    """Metashape-specific camera lookup, ray picking, and CRS projection."""

    georeferenced = True

    def __init__(self, config):
        if not Metashape.License().valid:
            raise RuntimeError("Invalid Metashape license. Activate Metashape first.")

        project_path = os.path.join(config.metashape_dir, "metashape.psx")
        if not os.path.isfile(project_path):
            raise FileNotFoundError(f"Metashape project not found: {project_path}")

        self.doc = Metashape.Document()
        self.doc.open(project_path)
        if not self.doc.chunks:
            raise RuntimeError(f"Metashape project has no chunks: {project_path}")
        self.chunk = self.doc.chunks[0]
        self.cameras = self.chunk.cameras
        self.crs = self.chunk.crs
        self.transform = self.chunk.transform.matrix

    def _camera_for_frame(self, frame_index: int, frame_path: str | None):
        if frame_path:
            stem = os.path.splitext(os.path.basename(frame_path))[0]
            camera = next((item for item in self.cameras if item.label == stem), None)
            if camera is not None:
                return camera
        # Enforce strict label matching to avoid projecting detections onto
        # a different camera when filenames and camera labels diverge.
        return None

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
        if depth_m is None or depth_m <= 0:
            return None

        distance = float(depth_m)

        camera = self._camera_for_frame(frame_index, frame_path)
        if camera is None or camera.transform is None:
            return None

        ray_point = camera.transform.mulp(
            camera.sensor.calibration.unproject(Metashape.Vector([x_px, y_px]))
        )

        camera_world = self.crs.project(self.transform.mulp(camera.center))
        ray_world = self.crs.project(self.transform.mulp(ray_point))
        camera_vec = np.asarray(camera_world[:3], dtype=np.float64)
        ray_vec = np.asarray(ray_world[:3], dtype=np.float64)
        direction = ray_vec - camera_vec
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return None
        world_point = camera_vec + (direction / norm) * distance

        x, y, z = world_point
        return {
            "Image": camera.label,
            "x_px": float(x_px),
            "y_px": float(y_px),
            "X": float(x),
            "Y": float(y),
            "Z": float(z),
            "distance_from_camera": distance,
            "type": class_name,
            "id": detection_id,
            "confidence": float(confidence),
        }
