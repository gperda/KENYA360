import mapping
#import orthomosaic
#import meshing
#import detection
#import stereo
import animals_detection
import os
import shutil
import argparse
import signal
import subprocess
import sys
from config import Config
from depth import process_detected_frames
from reconstruction import create_workflow
from utils import logger, timeit

def main():
    # Initialize the configuration
    parser = argparse.ArgumentParser(description='360 Animals detection', usage='USAGE: main.py --config /path/to/config.ini')
    parser.add_argument('--config', type=str, default=None, help='Path to config.ini file')
    args = parser.parse_args()
    if args.config is not None:
        config = Config(args.config)
    else:
        exit(parser.usage)

    logger.info("Starting animals detection and mapping")

    execution_times = {}
    workflow = create_workflow(config)

    with timeit("main", execution_times):
        if os.path.exists(config.output_dir):
            shutil.rmtree(config.output_dir)
        os.makedirs(config.output_dir)

        # Stereo reconstruction
        with timeit("stereo.generate", execution_times):
            workflow.reconstruct(config)

        # # Cracks detection
        # if config.save_cracks_overlay:
        #     if os.path.exists(config.overlay_dir):
        #         shutil.rmtree(config.overlay_dir, ignore_errors=True)
        #     os.makedirs(config.overlay_dir, exist_ok=True)
        # if os.path.exists(config.detection_dir):
        #     shutil.rmtree(config.detection_dir, ignore_errors=True)
        # os.makedirs(config.detection_dir, exist_ok=True)

        # with timeit("detection.initialize", execution_times):
        #     detection.initialize(config)

        # with timeit("detection.detect", execution_times):
        #     detection.detect()

        #Animal detection folder
        if os.path.exists(config.signs_detection_dir):
            shutil.rmtree(config.signs_detection_dir, ignore_errors=True)
        os.makedirs(config.signs_detection_dir, exist_ok=True)

        try:
            with timeit("animals_detection.initialize", execution_times):
                animals_detection.initialize(config)

            with timeit("animals_detection.detect", execution_times):
                detection_result = animals_detection.detect(workflow)

            with timeit("depth.process_detected_frames", execution_times):
                process_detected_frames(
                    detection_result.frames,
                    detection_result.detected_frame_indices,
                    config,
                    detection_result=detection_result,
                )

            with timeit("animals_detection.write_positions_csv", execution_times):
                animals_detection.write_positions_csv(
                    detection_result,
                    workflow,
                    config.signs_positions_file,
                )

        # # Mesh creation
        # if os.path.exists(config.mesh_dir):
        #     shutil.rmtree(config.mesh_dir, ignore_errors=True)
        # os.makedirs(config.mesh_dir, exist_ok=True)

        # with timeit("meshing.create_mesh", execution_times):
        #     meshing.create_mesh(config, execution_times)
        
        # #Mesh texturing and orthomosaic generation
        # if os.path.exists(config.ortho_dir):
        #     shutil.rmtree(config.ortho_dir, ignore_errors=True)
        # os.makedirs(config.ortho_dir, exist_ok=True)

        # with timeit("orthomosaic.generate", execution_times):
        #     orthomosaic.generate(config, execution_times)

            if workflow.georeferenced:
                # Mapping in QGIS is only valid for georeferenced detections.
                if os.path.exists(config.maps_dir):
                    shutil.rmtree(config.maps_dir, ignore_errors=True)
                os.makedirs(config.maps_dir)


                with timeit("mapping.generate", execution_times):
                    mapping.generate(config, execution_times)
            else:
                logger.info("Skipping mapping: the selected workflow is not georeferenced.")
        finally:
            workflow.close()

    logger.info("\nExecution time summary:")
    for section, elapsed in execution_times.items():
        minutes, seconds = divmod(elapsed, 60)
        if int(minutes) == 0:
            logger.info(f"{section}: {seconds:.2f}s")
        else:
            logger.info(f"{section}: {int(minutes)}m {seconds:.2f}s")
    pass

if __name__ == "__main__":
    main()

