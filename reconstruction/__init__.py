def create_workflow(config):
    """Construct the trajectory workflow selected in configuration."""
    source = config.trajectory_source
    if source == "metashape":
        from reconstruction.metashape import MetashapeWorkflow

        workflow_type = MetashapeWorkflow
    elif source == "colmap":
        from reconstruction.colmap import ColmapWorkflow

        workflow_type = ColmapWorkflow
    elif source == "slam":
        from reconstruction.slam import SlamWorkflow

        workflow_type = SlamWorkflow
    else:
        raise ValueError(f"Unsupported trajectory source: {source}")
    return workflow_type(config)


__all__ = ["create_workflow"]
