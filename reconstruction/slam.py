from reconstruction.base import ReconstructionWorkflow


class SlamWorkflow(ReconstructionWorkflow):
    """Reserved SLAM adapter; no SLAM format or georeference is configured."""

    def __init__(self, config):
        raise NotImplementedError(
            "The SLAM trajectory workflow is not configured yet. Configure a SLAM "
            "trajectory format and an explicit transform to the mapping CRS before use."
        )

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
        raise NotImplementedError("SLAM detection projection is not implemented.")
