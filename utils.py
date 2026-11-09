import logging
from contextlib import contextmanager
import os
import shutil
import subprocess
import time
import numpy as np
#import pycolmap

logger = logging.getLogger("KENYA360")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def run_fastfoundation_depth(
    imgdir,
    frame_indices,
    metadata_file,
    output_dir,
    cameras,
    ckpt_dir=None,
    scale=0.5,
    valid_iters=32,
    z_far=200,
    num_gpus=1,
    conda_env=None,
    module_name="scripts.run_demo_batch",
    extra_args=None,
):
    """
    Run FastFoundation from a dedicated conda environment without installing any
    extra package into the current project environment.

    This wrapper intentionally does not import FastFoundation in-process.
    The target environment is expected to already contain the module and its
    requirements.
    """
    if not conda_env:
        logger.warning("FastFoundation conda environment is not configured; depth step skipped.")
        return None

    if not os.path.isdir(imgdir):
        raise FileNotFoundError(f"FastFoundation image directory does not exist: {imgdir}")
    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(f"FastFoundation metadata file does not exist: {metadata_file}")
    if not cameras:
        raise ValueError("At least one FastFoundation camera must be specified.")

    if isinstance(frame_indices, str):
        frame_indices_arg = frame_indices
    else:
        normalized_indices = sorted({int(idx) for idx in frame_indices})
        if not normalized_indices:
            raise ValueError("frame_indices must contain at least one frame index.")
        frame_indices_arg = ",".join(map(str, normalized_indices))

    os.makedirs(output_dir, exist_ok=True)

    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe is None:
        raise FileNotFoundError("conda executable not found; cannot invoke FastFoundation in its dedicated environment.")

    cmd = [
        conda_exe, "run", "-n", conda_env, "/home/ellie/miniforge3/envs/foundation_stereo/bin/python", "/media/ellie/Olympos/FoundationStereo/scripts/run_demo_batch.py",
        "--imgdir", os.fspath(imgdir),
        "--frames", frame_indices_arg,
        "--metadata_file", os.fspath(metadata_file),
        "--out_dir", os.fspath(output_dir),
        "--cameras", ",".join(cameras) if not isinstance(cameras, str) else cameras,
        "--scale", str(scale),
        "--ckpt_dir", str(ckpt_dir),
        "--valid_iters", str(valid_iters),
        "--z_far", str(z_far),
        "--num_gpus", str(num_gpus),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running FastFoundation via conda env '%s': %s", conda_env, " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FastFoundation failed with exit code {completed.returncode}. "
            f"stderr: {completed.stderr.strip() or 'no stderr output'}"
        )

    return output_dir

@contextmanager
def timeit(section, execution_times):
    start = time.time()
    logger.info(f"Started: {section}")
    yield
    elapsed = time.time() - start
    minutes, seconds = divmod(elapsed, 60)
    execution_times[section] = elapsed 
    if int(minutes) == 0:
        logger.info(f"{section} completed in {seconds:.2f}s")
    else:
        logger.info(f"{section} completed in {int(minutes)}m {seconds:.2f}s")

def setup_logger(name="ViatekApp", level=logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s][%(name)s] %(message)s', "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

#def euler_to_rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
# def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
#     """
#     Converts Euler angles (omega, phi, kappa) to a rotation matrix.

#     Args:
#         omega (float): Rotation around the X-axis in radians.
#         phi (float): Rotation around the Y-axis in radians.
#         kappa (float): Rotation around the Z-axis in radians.

#     Returns:
#         np.ndarray: 3x3 rotation matrix.
#     """
#     omega = np.radians(omega)
#     phi = np.radians(phi)
#     kappa = np.radians(kappa)

#     R_x = np.array([[1, 0, 0],
#                     [0, np.cos(omega), -np.sin(omega)],
#                     [0, np.sin(omega), np.cos(omega)]])
    
#     R_y = np.array([[np.cos(phi), 0, np.sin(phi)],
#                     [0, 1, 0],
#                     [-np.sin(phi), 0, np.cos(phi)]])
    
#     R_z = np.array([[np.cos(kappa), -np.sin(kappa), 0],
#                     [np.sin(kappa), np.cos(kappa), 0],
#                     [0, 0, 1]])
    
#     return R_z @ R_x @ R_y

def quaternion_to_rotation_matrix(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_focal_and_distortion(cam):
    """Extract focal length and up to two distortion coefficients."""
    if cam.model == "SIMPLE_PINHOLE":
        return cam.params[0], 0.0, 0.0
    elif cam.model == "PINHOLE":
        return cam.params[0], 0.0, 0.0
    elif cam.model == "SIMPLE_RADIAL":
        return cam.params[0], cam.params[1], 0.0
    elif cam.model == "RADIAL":
        return cam.params[0], cam.params[1], cam.params[2]
    else:
        raise NotImplementedError(f"Unsupported camera model: {cam.model}")


def euler_to_rotation_matrix(roll, pitch, yaw):
    return Metashape.Utils.ypr2mat(Metashape.Vector([yaw, roll, pitch+100]))

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Converts Euler angles (omega, phi, kappa) to a rotation matrix.

    Args:
        omega (float): Rotation around the X-axis in radians.
        phi (float): Rotation around the Y-axis in radians.
        kappa (float): Rotation around the Z-axis in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def read_trajectory_txt(file_path: str, separator: str) -> list:
    """
    Reads a GNSS trajectory file in TXT format.

    Args:
        file_path (str): Path to the trajectory TXT file.

    Returns:
        list: List of dictionaries with keys 'name', 'position', 'rotation', and 'angles'.
    """
    cameras = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines
            data = line.split(separator)
            name = data[0].strip()
            x, y, z = map(np.float64, data[1:4])
            #omega, phi, kappa = map(np.float64, data[4:7])
            roll, pitch, yaw = map(np.float64, data[4:7])
            #rotation_matrix = euler_to_rotation_matrix(omega, phi, kappa)
            rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
            cameras.append({
                "name": name,
                "position": np.array([x, y, z]),
                "rotation": np.array(rotation_matrix),
                "angles": np.array([roll, pitch, yaw])
            })
    # Close the file after reading
    file.close()
    return cameras

def read_orientation_txt(file_path:str, separator: str) -> list:
    """
    Reads a GNSS trajectory file in TXT format.

    Args:
        file_path (str): Path to the trajectory TXT file.

    Returns:
        list: List of dictionaries with keys 'name', 'position', 'rotation', and 'angles'.
    """
    cameras = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines
            data = line.split(separator)
            name = data[0].strip()
            x, y, z = map(np.float64, data[1:4])
            omega, phi, kappa = map(np.float64, data[4:7])
            r11, r12, r13 = map(np.float64, data[7:10])
            r21, r22, r23 = map(np.float64, data[10:13])
            r31, r32, r33 = map(np.float64, data[13:16])
            cameras.append({
                "name": name,
                "position": np.array([x, y, z]),
                "rotation": np.array([[r11, r12, r13],
                                      [r21, r22, r23],
                                      [r31, r32, r33]]),
                "angles": np.array([omega, phi, kappa])
            })
    file.close()
    return cameras


def read_reconstruction_colmap(file_path: str) -> list:
    """
    Reads a COLMAP reconstruction object and extracts camera positions and rotations.

    Args:
        reconstruction (pycolmap.Reconstruction): COLMAP reconstruction object.

    Returns:
        list: List of dictionaries with keys 'name', 'position', 'rotation', and 'angles'.
    """

    reconstruction = pycolmap.Reconstruction(file_path)
    cameras = []
    for image_id, image in reconstruction.images.items():
        name = image.name
        position = image.projection_center()
        rotation_matrix = image.cam_from_world.rotation.matrix()
        omega, phi, kappa = np.zeros(3)  # Placeholder, as COLMAP does not directly provide these angles
        
        cameras.append({
            "name": name,
            "position": np.array(position),
            "rotation": np.array(rotation_matrix),
            "angles": np.array([omega, phi, kappa])  # Optional: Compute angles if needed
        })
    return cameras


def export_reconstruction_to_bundler(reconstruction_path):
    """
    Reads a COLMAP reconstruction and exports it in Bundler format.

    Args:
        reconstruction (pycolmap.Reconstruction): COLMAP reconstruction object.

    Returns:
        string: bundler file name
    """
    reconstruction = pycolmap.Reconstruction(reconstruction_path)

    images = list(reconstruction.images.values())
    points3D = list(reconstruction.points3D.values())

    with open(config.bundle_out_file, 'w') as f:
        f.write("# Bundle file v0.3\n")
        f.write(f"{len(images)} {len(points3D)}\n")

        for image in images:
            cam = reconstruction.cameras[image.camera_id]
            f_length, k1, k2 = get_focal_and_distortion(cam)

            f.write(f"{f_length} {k1} {k2}\n")

            R = quaternion_to_rotation_matrix(image.qvec)
            for row in R:
                f.write(' '.join(map(str, row)) + "\n")
            f.write(' '.join(map(str, image.tvec)) + "\n")

        for point in points3D:
            xyz = point.xyz
            rgb = point.color
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n {rgb[0]} {rgb[1]} {rgb[2]}\n")

            track = point.track
            f.write(f"{len(track)}")
            for obs in track:
                image_idx = list(reconstruction.images.keys()).index(obs.image_id)
                x, y = reconstruction.images[obs.image_id].xys[obs.point2D_idx]
                f.write(f" {image_idx} {obs.point2D_idx} {x} {y}")
            f.write("\n")
    
def format_optimized_trajectory(input):
    # Remove first line, replace "\t" with ","
    # Copy only first 7 columns, remove spaces in new first line
    with open(input, 'r') as f:
        lines = f.readlines()
    with open(input, 'w') as f:
        for i, line in enumerate(lines[1:]):
            parts = line.strip().split('\t')
            formatted_line = ','.join(parts)
            if i == 0:
                formatted_line = formatted_line.replace(' ', '')
            f.write(formatted_line + '\n')
    

logger = setup_logger()