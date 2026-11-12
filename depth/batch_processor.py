import glob
import os

import cv2
import numpy as np

from reconstruction.base import normalize_depth_to_meters, sample_depth_at_bbox_center
from utils import logger, run_fs_depth


def _find_depth_paths(frame_name, depth_root):
    stem, _ = os.path.splitext(os.path.basename(frame_name or ""))
    if not stem:
        return []

    patterns = [
        os.path.join(depth_root, f"{stem}*.jpg"),
        os.path.join(depth_root, f"{stem}*.npy"),
        os.path.join(depth_root, f"{stem}*.tif"),
        os.path.join(depth_root, f"{stem}*.tiff"),
        os.path.join(depth_root, f"{stem}*.exr"),
        os.path.join(depth_root, f"*{stem}*.png"),
        os.path.join(depth_root, f"*{stem}*.npy"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    # Most relevant results are the exact stems first; keep a stable deduplicated order.
    unique = []
    seen = set()
    for path in sorted(matches):
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _load_depth_map(depth_path):
    if depth_path is None or not os.path.exists(depth_path):
        return None

    _, extension = os.path.splitext(depth_path.lower())
    if extension == ".npy":
        array = np.load(depth_path)
    else:
        array = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if array is None:
            array = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        array = array[..., 0]
    return array


def process_detected_frames(frames, frame_indices, config, detection_result=None):
    """Run FoundationStereo depth extraction and compute a metric depth distance for each detection."""
    depth_source = getattr(config, "depth_source", "metashape")
    if depth_source == "metashape":
        return {}
    if depth_source != "foundation_stereo":
        logger.warning("Depth source '%s' is not implemented; skipping depth processing.", depth_source)
        return {}

    conda_env = getattr(config, "fs_conda_env", None)
    if not conda_env:
        logger.warning("FoundationStereo depth source is selected but no conda env is configured.")
        return {}

    module_name = getattr(config, "fs_module", "scripts.run_demo_batch")
    output_root = getattr(config, "depth_dir")
    os.makedirs(output_root, exist_ok=True)
    if not frame_indices:
        logger.info("No detected frames; skipping FoundationStereo batch depth processing.")
        return {}

    def setting(name, default):
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                value = getter("General", name, fallback=None)
            except TypeError:
                value = None
            if value is not None and str(value).strip():
                return value
        return default

    input_root = getattr(config, "input_dir", config.frames_dir)
    imgdir = input_root
    metadata_file = setting(
        "fs_metadata_file",
        os.path.join(input_root, "metadata.json"),
    )
    camera_default = getattr(config, "camera", os.path.basename(config.frames_dir.rstrip(os.sep)))
    cameras = setting("fs_cameras", camera_default)
    if isinstance(cameras, str):
        cameras = [camera.strip() for camera in cameras.split(",") if camera.strip()]

    try:
        run_fs_depth(
            imgdir=imgdir,
            frame_indices=frame_indices,
            metadata_file=metadata_file,
            output_dir=output_root,
            cameras=cameras,
            ckpt_dir=setting("fs_ckpt_dir", None),
            scale=float(setting("fs_scale", 0.5)),
            valid_iters=int(setting("fs_valid_iters", 32)),
            z_far=float(setting("fs_z_far", 200)),
            num_gpus=int(setting("fs_num_gpus", 1)),
            conda_env=conda_env,
            module_name=module_name,
        )
    except Exception as exc:
        logger.warning("FoundationStereo batch depth invocation failed: %s", exc)
        return {}

    if detection_result is None or not getattr(detection_result, "frame_masks", None):
        logger.info("No detection masks available; skipping depth-based distance extraction.")
        return {}

    debug_dir = getattr(config, "depth_overlay_dir")
    os.makedirs(debug_dir, exist_ok=True)
    z_far = float(getattr(config, "fs_z_far", 200.0))
    frame_index_to_name = {}
    for idx, frame in enumerate(frames):
        if isinstance(frame, str):
            frame_index_to_name[idx] = os.path.basename(frame)
        else:
            frame_index_to_name[idx] = f"{idx}.jpg"

    depth_by_frame = {}
    for frame_idx in sorted(frame_indices):
        frame_masks = detection_result.frame_masks.get(frame_idx, [])
        if not frame_masks:
            continue

        frame_name = frame_index_to_name.get(frame_idx, f"{frame_idx}.jpg")
        depth_paths = _find_depth_paths(frame_name, output_root)
        if not depth_paths:
            logger.warning("No depth map found for frame %s; skipping depth extraction.", frame_idx)
            continue

        depth_map = _load_depth_map(depth_paths[0])
        if depth_map is None:
            logger.warning("Depth map is unreadable for frame %s: %s", frame_idx, depth_paths[0])
            continue

        debug_img = None
        original_frame = frames[frame_idx] if frame_idx < len(frames) else None
        if isinstance(original_frame, str) and os.path.exists(original_frame):
            debug_img = cv2.imread(depth_paths[0], cv2.IMREAD_COLOR)
        elif isinstance(original_frame, np.ndarray):
            debug_img = original_frame.copy()
        if debug_img is None:
            debug_img = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

        frame_depth_values = {}
        for detection in frame_masks:
            x_center = float(detection.get("x_center", 0.0))
            y_center = float(detection.get("y_center", 0.0))
            mask = detection.get("mask")
            raw_depth = sample_depth_at_bbox_center(depth_map, x_center, y_center, mask)
            if raw_depth is None:
                continue
            metric_depth = normalize_depth_to_meters(raw_depth, z_far=z_far)
            if metric_depth is None:
                continue

            detection_id = str(detection.get("detection_id", ""))
            frame_depth_values[detection_id] = metric_depth
            detection["distance_m"] = metric_depth

            bbox = detection.get("bbox", (x_center, y_center, 0, 0))
            x1, y1, w, h = bbox
            x2 = int(round(float(x1) + float(w)))
            y2 = int(round(float(y1) + float(h)))
            cv2.rectangle(debug_img, (int(round(float(x1))), int(round(float(y1)))), (x2, y2), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(round(x_center)), int(round(y_center))), 4, (0, 0, 255), -1)
            cv2.putText(
                debug_img,
                f"{metric_depth:.2f}m",
                (int(round(x_center)) + 8, int(round(y_center)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        depth_by_frame[frame_idx] = frame_depth_values
        output_name = os.path.splitext(os.path.basename(frame_name))[0] + ".png"
        cv2.imwrite(os.path.join(debug_dir, output_name), debug_img)

    return depth_by_frame
