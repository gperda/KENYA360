import mapping
#import orthomosaic
#import meshing
#import detection
#import stereo
import animals_detection
import os
import shutil
import Metashape
import argparse
import signal
import subprocess
import sys
from config import Config
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

    if not Metashape.License().valid:
        exit("Invalid Metashape license. Activate Metashape first.")

    logger.info("Starting animals detection and mapping")

    execution_times = {}

    with timeit("main", execution_times):
        if os.path.exists(config.output_dir):
            shutil.rmtree(config.output_dir)
        os.makedirs(config.output_dir)

        # # Stereo reconstruction
        # if os.path.exists(config.metashape_dir):
        #     shutil.rmtree(config.metashape_dir, ignore_errors=True)
        # os.makedirs(config.metashape_dir)

        # with timeit("stereo.generate", execution_times):
        #     stereo.generate(config)

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

        # Traffic signs detection
        if os.path.exists(config.signs_detection_dir):
            shutil.rmtree(config.signs_detection_dir, ignore_errors=True)
        os.makedirs(config.signs_detection_dir, exist_ok=True)

        with timeit("animals_detection.initialize", execution_times):
            animals_detection.initialize(config)

        with timeit("signs_detection.detect", execution_times):
            animals_detection.detect()

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

        # Mapping in QGIS
        if os.path.exists(config.maps_dir):
            shutil.rmtree(config.maps_dir, ignore_errors=True)
        os.makedirs(config.maps_dir)

        with timeit("mapping.generate", execution_times):
            mapping.generate(config, execution_times)

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

