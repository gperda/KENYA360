from reconstruction.base import ReconstructionWorkflow


class ColmapWorkflow(ReconstructionWorkflow):
    """Reserved COLMAP adapter; dataset and georeferencing conventions are not configured."""

    def __init__(self, config):
        raise NotImplementedError(
            "The COLMAP trajectory workflow is not configured yet. Supply a COLMAP "
            "reconstruction format and an explicit transform to the mapping CRS before use."
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
        raise NotImplementedError("COLMAP detection projection is not implemented.")
