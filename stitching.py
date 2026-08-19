import os
from dataclasses import dataclass

import cv2
import numpy as np
import yaml


def _create_opencv_stitcher():
    if hasattr(cv2, "Stitcher_create"):
        return cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    if hasattr(cv2, "createStitcher"):
        return cv2.createStitcher(False)
    raise RuntimeError("OpenCV Stitcher API is not available in this cv2 build")


@dataclass
class StitchCalibration:
    K0: np.ndarray
    d0: np.ndarray
    K1: np.ndarray
    d1: np.ndarray
    R_0_to_1: np.ndarray
    T_0_to_1: np.ndarray


def _build_intrinsics(intrinsics):
    fx, fy, cx, cy = intrinsics
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _find_calibration_file(input_dir):
    candidates = [
        os.path.join(input_dir, "camchain.yaml"),
        os.path.join(input_dir, "out", "camchain.yaml"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find camchain calibration file in input_dir or input_dir/out"
    )


def load_camchain(calibration_file):
    with open(calibration_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "cam0" not in data or "cam1" not in data:
        raise ValueError("Calibration file must contain cam0 and cam1 sections")

    cam0 = data["cam0"]
    cam1 = data["cam1"]

    if "intrinsics" not in cam0 or "distortion_coeffs" not in cam0:
        raise ValueError("cam0 intrinsics/distortion_coeffs are missing")
    if "intrinsics" not in cam1 or "distortion_coeffs" not in cam1:
        raise ValueError("cam1 intrinsics/distortion_coeffs are missing")
    if "T_cn_cnm1" not in cam1:
        raise ValueError("cam1 T_cn_cnm1 is missing")

    K0 = _build_intrinsics(cam0["intrinsics"])
    K1 = _build_intrinsics(cam1["intrinsics"])
    d0 = np.asarray(cam0["distortion_coeffs"], dtype=np.float64).reshape(-1)
    d1 = np.asarray(cam1["distortion_coeffs"], dtype=np.float64).reshape(-1)

    T = np.asarray(cam1["T_cn_cnm1"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_cn_cnm1 must be a 4x4 matrix")

    R_0_to_1 = T[:3, :3]
    T_0_to_1 = T[:3, 3]

    return StitchCalibration(K0=K0, d0=d0, K1=K1, d1=d1, R_0_to_1=R_0_to_1, T_0_to_1=T_0_to_1)
def rectify_pair(img0, img1, calib: StitchCalibration, alpha=0):
    """Undistort + rectify both images so epipolar lines are horizontal rows.
 
    alpha: 0 = crop to valid pixels only (no black borders, some FOV lost)
           1 = keep full FOV (black borders included)
    """
    h, w = img0.shape[:2]
    image_size = (w, h)
 
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        calib.K0, calib.d0,
        calib.K1, calib.d1,
        image_size,
        calib.R_0_to_1, calib.T_0_to_1,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
 
    map0x, map0y = cv2.initUndistortRectifyMap(calib.K0, calib.d0, R0, P0, image_size, cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(calib.K1, calib.d1, R1, P1, image_size, cv2.CV_32FC1)
 
    rect0 = cv2.remap(img0, map0x, map0y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
 
    rectify_data = {
        "R0": R0, "R1": R1, "P0": P0, "P1": P1, "Q": Q,
        "roi0": roi0, "roi1": roi1,
        "map0x": map0x, "map0y": map0y,
        "map1x": map1x, "map1y": map1y,
    }
    return rect0, rect1, rectify_data
 
 
def crop_to_common_roi(rect0, rect1, roi0, roi1):
    """After rectification with alpha=0, cv2.stereoRectify gives you the
    valid (no black-border) region for each image as (x, y, w, h). Crop
    both images to the INTERSECTION of the two ROIs so we're merging the
    same physical region from both cameras.
    """
    x0, y0, w0, h0 = roi0
    x1, y1, w1, h1 = roi1
 
    x = max(x0, x1)
    y = max(y0, y1)
    x_end = min(x0 + w0, x1 + w1)
    y_end = min(y0 + h0, y1 + h1)
 
    if x_end <= x or y_end <= y:
        raise ValueError("Rectified images have no overlapping valid region")
 
    common_roi = (x, y, x_end - x, y_end - y)
    return rect0[y:y_end, x:x_end], rect1[y:y_end, x:x_end], common_roi
 
 
def blend_rectified(rect0, rect1, weight0=0.5):
    """Directly blend two rectified, ROI-cropped images. No disparity is
    used, so this is only geometrically exact for scene content at
    infinity -- nearer content will show horizontal ghosting (see module
    docstring). weight0 controls the left/right blend ratio (0.5 = even).
    """
    out = (rect0.astype(np.float32) * weight0
           + rect1.astype(np.float32) * (1 - weight0))
    return np.clip(out, 0, 255).astype(np.uint8)


def stitch_with_opencv_stitcher(img0, img1):
    stitcher = _create_opencv_stitcher()
    status, stitched = stitcher.stitch([img0, img1])

    if status != cv2.Stitcher_OK or stitched is None:
        return None, {
            "backend": "opencv_stitcher",
            "status": int(status),
            "error": f"OpenCV Stitcher failed with status code {int(status)}",
        }

    metadata = {
        "backend": "opencv_stitcher",
        "status": int(status),
        "stitched_shape": stitched.shape,
        "input_shape_cam0": img0.shape,
        "input_shape_cam1": img1.shape,
    }
    return stitched, metadata
 
 
def merge_stereo_pair(
    img0,
    img1,
    calib: StitchCalibration = None,
    alpha=0,
    weight0=0.5,
    backend="rectify_map",
):
    """Full pipeline: rectify -> crop to common ROI -> direct blend.
 
    No disparity computation. Returns the merged image plus the
    intermediate rectified/cropped images and rectification data.
    """
    if backend == "opencv_stitcher":
        return stitch_with_opencv_stitcher(img0, img1)

    if backend != "rectify_map":
        raise ValueError(f"Unsupported stitching backend: {backend}")

    if calib is None:
        raise ValueError("calib is required when backend is 'rectify_map'")

    rect0, rect1, rectify_data = rectify_pair(img0, img1, calib, alpha=alpha)
    crop0, crop1, common_roi = crop_to_common_roi(rect0, rect1, rectify_data["roi0"], rectify_data["roi1"])
    merged = blend_rectified(crop0, crop1, weight0=weight0)
 
    metadata = {
        "backend": "rectify_map",
        "rectify_data": rectify_data,
        "common_roi": common_roi,
        "rect0": rect0,
        "rect1": rect1,
        "crop0": crop0,
        "crop1": crop1,
    }
    return merged, metadata



def remap_stitched_point_to_source(x, y, metadata, preferred_camera="cam1"):
    if metadata.get("backend") != "rectify_map":
        return None

    h, w = metadata["crop0"].shape[:2]
    px = int(round(x))
    py = int(round(y))

    if px < 0 or py < 0 or px >= w or py >= h:
        return None

    crop_x, crop_y, _, _ = metadata.get("common_roi", (0, 0, w, h))
    rx = px + int(crop_x)
    ry = py + int(crop_y)

    rectify_data = metadata["rectify_data"]
    map0x = rectify_data["map0x"]
    map0y = rectify_data["map0y"]
    map1x = rectify_data["map1x"]
    map1y = rectify_data["map1y"]
    src_h, src_w = map0x.shape

    def _sample_source(mx, my):
        sx = float(mx[ry, rx])
        sy = float(my[ry, rx])
        if not np.isfinite(sx) or not np.isfinite(sy):
            return None
        if sx < 0.0 or sy < 0.0 or sx >= src_w or sy >= src_h:
            return None
        return sx, sy

    cam0_src = _sample_source(map0x, map0y)
    cam1_src = _sample_source(map1x, map1y)

    if cam0_src is None and cam1_src is None:
        return None

    if preferred_camera == "cam0":
        if cam0_src is not None:
            return "cam0", cam0_src[0], cam0_src[1]
        return "cam1", cam1_src[0], cam1_src[1]

    if cam1_src is not None:
        return "cam1", cam1_src[0], cam1_src[1]
    return "cam0", cam0_src[0], cam0_src[1]


def resolve_calibration_file(input_dir):
    return _find_calibration_file(input_dir)
