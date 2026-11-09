import os
import cv2
import glob
import csv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
)
import numpy as np
import warnings
import shutil
from dataclasses import dataclass, field
from utils import logger
from reconstruction.base import normalize_depth_to_meters, sample_depth_at_bbox_center
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction
# from sahi.utils.cv import read_image_as_pil
HEIGHT = 640
WIDTH = 640

model = None
config = None
_depth_map_cache = {}
_depth_lookup_root = None


def _depth_root_dir():
    if config is None:
        return None
    if str(getattr(config, "depth_source", "metashape")).lower() != "fastfoundation":
        return None
    depth_root = getattr(config, "depth_dir", None)
    if not depth_root or not os.path.isdir(depth_root):
        return None
    return depth_root


def _depth_z_far_meters() -> float:
    getter = getattr(config, "get", None)
    if callable(getter):
        value = getter("General", "fastfoundation_z_far", fallback=None)
        if value is not None and str(value).strip():
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float(getattr(config, "fastfoundation_z_far", 200.0))


def _find_depth_map_path(frame_name: str, depth_root: str) -> str | None:
    stem, _ = os.path.splitext(os.path.basename(frame_name or ""))
    if not stem:
        return None

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
    if not matches:
        return None

    unique = []
    seen = set()
    for path in sorted(matches):
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique[0] if unique else None


def _load_depth_map(depth_path: str):
    if depth_path is None or not os.path.exists(depth_path):
        return None

    _, extension = os.path.splitext(depth_path.lower())
    if extension == ".npy":
        depth_array = np.load(depth_path)
    else:
        depth_array = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_array is None:
            depth_array = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_array is None:
        return None

    depth_array = np.asarray(depth_array, dtype=np.float32)
    if depth_array.ndim == 3 and depth_array.shape[-1] in (1, 3, 4):
        depth_array = depth_array[..., 0]
    if depth_array.ndim != 2:
        return None
    return depth_array


def _depth_map_for_frame(frame_name: str):
    depth_root = _depth_root_dir()
    if depth_root is None:
        return None

    global _depth_lookup_root
    if _depth_lookup_root != depth_root:
        _depth_map_cache.clear()
        _depth_lookup_root = depth_root

    stem, _ = os.path.splitext(os.path.basename(frame_name or ""))
    if not stem:
        return None
    if stem in _depth_map_cache:
        return _depth_map_cache[stem]

    depth_path = _find_depth_map_path(frame_name, depth_root)
    depth_map = _load_depth_map(depth_path) if depth_path else None
    _depth_map_cache[stem] = depth_map
    return depth_map


def _depth_m_for_detection(frame_name: str, x_px: float, y_px: float, mask=None) -> float | None:
    depth_map = _depth_map_for_frame(frame_name)
    if depth_map is None:
        return None

    raw_depth = sample_depth_at_bbox_center(depth_map, x_px, y_px, mask)
    return normalize_depth_to_meters(raw_depth, z_far=_depth_z_far_meters())


@dataclass
class DetectionRecord:
    frame_index: int
    frame_name: str
    frame_path: str | None
    detection_id: int | str
    class_name: str
    x_center: float
    y_center: float
    bbox: tuple[float, float, float, float]
    mask: object | None
    confidence: float
    depth_m: float | None = None
    projection: dict | None = None


@dataclass
class DetectionResult:
    frames: list
    detected_frame_indices: set[int]
    frame_masks: dict[int, list[dict]] = field(default_factory=dict)
    records: list[DetectionRecord] = field(default_factory=list)

def visualize_frame_output(frame_idx, video_frames, outputs):
    img = load_frame(video_frames[frame_idx])
    img = img.copy()  # avoid modifying original

    annotator = Annotator(img, line_width=1)

    for i in range(len(outputs["out_probs"])):
        box_xywh = outputs["out_boxes_xywh"][i]
        prob = outputs["out_probs"][i]
        obj_id = outputs["out_obj_ids"][i]
        binary_mask = outputs["out_binary_masks"][i]

        # Convert XYWH -> XYXY
        x, y, w, h = rel_to_abs_coords([box_xywh], img.shape[1], img.shape[0], coord_type="box")[0]
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        color = colors(obj_id, True)
        label = f"{obj_id}"

        # Draw bounding box
        annotator.box_label([x1, y1, x2, y2], label, color=color)

        # Draw mask (if available)
        if binary_mask is not None:
            mask = binary_mask.astype(bool)
            annotator.masks(
                np.array([mask], dtype=np.uint8),
                colors=[color]
            )

    result = annotator.result()

    return result

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

def rel_to_abs_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="box"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
    coords: List of coordinates
    coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x * IMG_WIDTH, y * IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x * IMG_WIDTH, y * IMG_HEIGHT, w * IMG_WIDTH, h * IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

# def detect_animals_sahi():
#     detection_model = AutoDetectionModel.from_pretrained(
#         model_type='ultralytics',
#         model_path=config.model_path,
#         confidence_threshold=0.4,
#         device="cuda:0"
#     )

    

#     for image_path in glob.glob(os.path.join(config.frames_dir, "*.jpg")):
#         image = read_image_as_pil(image_path)
#         prediction = get_sliced_prediction(
#             image=image,
#             detection_model=detection_model,
#             slice_height=256,
#             slice_width=256,
#             overlap_height_ratio=0.2,
#             overlap_width_ratio=0.2,
#         )
#         prediction.export_visuals(config.signs_detection_dir, os.path.basename(image_path))

# def detect_animals_sam3(chunk, f) -> np.ndarray:
#     """
#     Detects and annotates animals in an image using SAM3 video predictor.
#     Args:
#         img (np.ndarray): Input image.
#         chunk: Metashape chunk with camera poses.
#         camera: Metashape camera of detection result.
#         f: output file.
#     Returns:
#         detected_img (np.ndarray): Image with detected animals annotated.
#     """

#     # load "video_frames_for_vis" for visualization purposes (they are not used by the model)
#     video_path = config.frames_dir
#     if isinstance(video_path, str) and video_path.endswith(".mp4"):
#         cap = cv2.VideoCapture(video_path)
#         video_frames_for_vis = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         cap.release()
#     else:
#         video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
#         try:
#             # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
#             video_frames_for_vis.sort(
#                 key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
#             )
#         except ValueError:
#             # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
#             print(
#                 f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
#                 f"falling back to lexicographic sort."
#             )
#             video_frames_for_vis.sort()

#     crs = chunk.crs
#     T = chunk.transform.matrix
#     cameras = chunk.cameras

#     prompts = ["zebra", "rhino", "warthog", "elephant"]

#     video_predictor = build_sam3_video_predictor()

#     response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=config.frames_dir,
#         half=config.sam3_half,
#         )
#     )
#     session_id = response["session_id"]

#     for prompt in prompts:
#         response = video_predictor.handle_request(
#             request=dict(
#                 type="add_prompt",
#                 session_id=session_id,
#                 frame_index=1,
#                 text=prompt,
#             )
#         )
#         out = response["outputs"]
#         outputs_per_frame = propagate_in_video(video_predictor, session_id)
        
#         for frame_idx in range(0, len(outputs_per_frame) - 1):
#             frame_outputs = outputs_per_frame[frame_idx]
#             if (frame_outputs["out_obj_ids"].shape[0] > 0):            
#                 detected_img = visualize_frame_output(
#                     frame_idx,
#                     video_frames_for_vis,
#                     outputs=frame_outputs,
#                 )
#                 cv2.imwrite(
#                     os.path.join(config.signs_detection_dir, f"{prompt}_{frame_idx}.jpg"),
#                     cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR)
#                 )
            
#             if frame_idx > len(cameras) - 1:
#                 continue
#             camera = cameras[frame_idx]
#             if camera.transform is None:
#                 continue
            
#             for j in range(0, len(frame_outputs["out_obj_ids"])):
#                 hi, wi = frame_outputs["out_binary_masks"].shape[1:3]
#                 box_xywh = frame_outputs["out_boxes_xywh"][j]
#                 prob = frame_outputs["out_probs"][j]
#                 obj_id = frame_outputs["out_obj_ids"][j]
#                 x, y, w, h = rel_to_abs_coords([box_xywh], wi, hi, coord_type="box")[0]
#                 x_center = x + w / 2
#                 y_center = y + h / 2
#                 class_name = prompt  # use the prompt as the class name
                
#                 pt_3D = chunk.tie_points.pickPoint(
#                     camera.center,
#                     camera.transform.mulp(
#                         camera.sensor.calibration.unproject(Metashape.Vector([x_center, y_center]))
#                     ),
#                 )
#                 pt_3D_world = crs.project(T.mulp(pt_3D))
#                 proj_xy = crs.project(T.mulp(camera.center))
#                 distance = np.linalg.norm(pt_3D_world[:2] - proj_xy[:2])
#                 X, Y, Z = pt_3D_world
#                 line = f"{camera.label},{x_center:.2f},{y_center:.2f},{X:.6f},{Y:.6f},{Z:.6f},{distance},{class_name},{obj_id},{prob:.4f}\n"
#                 f.write(line)

def detect_animals_sam3(prompts) -> DetectionResult:
    # Load video frames for visualization
    video_path = config.frames_dir
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()

    frame_output_names = {}
    combined_output_frames = {}
    detected_frame_indices = set()
    frame_masks = {}
    records = []

    for frame_idx, frame_ref in enumerate(video_frames_for_vis):
        if isinstance(frame_ref, str):
            frame_output_names[frame_idx] = os.path.basename(frame_ref)
        else:
            frame_output_names[frame_idx] = f"{frame_idx}.jpg"

    # Split frames into blocks — done once, outside all loops
    BLOCK_SIZE = 100
    total_frames = len(video_frames_for_vis)
    frame_blocks = [
        video_frames_for_vis[i:i + BLOCK_SIZE]
        for i in range(0, total_frames, BLOCK_SIZE)
    ]

    video_predictor = build_sam3_video_predictor()

    for prompt in prompts:
        # --- Process all blocks for this prompt ---
        for block_idx, frame_block in enumerate(frame_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_end = block_start + len(frame_block)
            print(f"[{prompt}] Processing block {block_idx + 1}/{len(frame_blocks)}: frames {block_start}–{block_end - 1}")

            # Write block frames to a temp directory
            block_dir = os.path.join(config.frames_dir, f"_block_{block_idx:04d}")
            os.makedirs(block_dir, exist_ok=True)
            for local_idx, frame in enumerate(frame_block):
                if isinstance(frame, str):
                    dst = os.path.join(block_dir, f"{local_idx}.jpg")
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(frame), dst)
                else:
                    cv2.imwrite(
                        os.path.join(block_dir, f"{local_idx}.jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    )

            # Open a session for this prompt + block
            response = video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=block_dir,
                    half=config.sam3_half,
                )
            )
            session_id = response["session_id"]

            # Add the text prompt on frame 0 of this block
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt,
                )
            )

            # Propagate through the block
            outputs_per_frame = propagate_in_video(video_predictor, session_id)

            for local_frame_idx in range(0, len(frame_block)):
                frame_outputs = outputs_per_frame[local_frame_idx]
                global_frame_idx = block_start + local_frame_idx

                if frame_outputs["out_obj_ids"].shape[0] > 0:
                    detected_frame_indices.add(global_frame_idx)
                    print("added frame index", global_frame_idx, frame_output_names[global_frame_idx])

                    detected_img = visualize_frame_output(
                        global_frame_idx,
                        video_frames_for_vis,
                        outputs=frame_outputs,
                    )
                    if global_frame_idx not in combined_output_frames:
                        if isinstance(video_frames_for_vis[global_frame_idx], str):
                            combined_output_frames[global_frame_idx] = load_frame(video_frames_for_vis[global_frame_idx]).copy()
                        else:
                            combined_output_frames[global_frame_idx] = video_frames_for_vis[global_frame_idx].copy()

                    base_frame = load_frame(video_frames_for_vis[global_frame_idx]) if isinstance(video_frames_for_vis[global_frame_idx], str) else video_frames_for_vis[global_frame_idx]
                    changed_pixels = np.any(detected_img != base_frame, axis=2)
                    combined_output_frames[global_frame_idx][changed_pixels] = detected_img[changed_pixels]
                    
                    

                    for j in range(len(frame_outputs["out_obj_ids"])):
                        hi, wi = frame_outputs["out_binary_masks"].shape[1:3]
                        box_xywh = frame_outputs["out_boxes_xywh"][j]
                        prob = frame_outputs["out_probs"][j]
                        obj_id = frame_outputs["out_obj_ids"][j]
                        x, y, w, h = rel_to_abs_coords([box_xywh], wi, hi, coord_type="box")[0]
                        x_center = x + w / 2
                        y_center = y + h / 2
                        class_name = prompt
                        mask = frame_outputs["out_binary_masks"][j].astype(bool)
                        frame_masks.setdefault(global_frame_idx, []).append(
                            {
                                "frame_index": global_frame_idx,
                                "frame_name": frame_output_names[global_frame_idx],
                                "detection_id": int(obj_id) if isinstance(obj_id, (int, np.integer)) else str(obj_id),
                                "class_name": class_name,
                                "x_center": float(x_center),
                                "y_center": float(y_center),
                                "bbox": (float(x), float(y), float(w), float(h)),
                                "mask": mask,
                                "confidence": float(prob),
                            }
                        )

                        frame_ref = video_frames_for_vis[global_frame_idx]
                        frame_path = frame_ref if isinstance(frame_ref, str) else None
                        depth_frame_name = frame_path or frame_output_names[global_frame_idx]
                        detection_id = int(obj_id) if isinstance(obj_id, (int, np.integer)) else str(obj_id)
                        records.append(
                            DetectionRecord(
                                frame_index=global_frame_idx,
                                frame_name=frame_output_names[global_frame_idx],
                                frame_path=frame_path,
                                detection_id=detection_id,
                                class_name=class_name,
                                x_center=float(x_center),
                                y_center=float(y_center),
                                bbox=(float(x), float(y), float(w), float(h)),
                                mask=mask,
                                confidence=float(prob),
                                depth_m=_depth_m_for_detection(depth_frame_name, x_center, y_center, mask),
                            )
                        )
            shutil.rmtree(block_dir, ignore_errors=True)
            # Close session at the end of each block, still within the prompt loop
            video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

    for frame_idx, combined_img in combined_output_frames.items():
        output_name = frame_output_names[frame_idx]
        cv2.imwrite(
            os.path.join(config.signs_detection_dir, output_name),
            cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR),
        )

    return DetectionResult(video_frames_for_vis, detected_frame_indices, frame_masks, records)

def detect_animals_yolo(img, frame_index, frame_path) -> tuple[np.ndarray, bool, list[dict], list[DetectionRecord]]:
    """
    Detects and annotates traffic signs in an image using YOLO model.
    Args:
        img (np.ndarray): Input image.
        chunk: Metashape chunk with camera poses.
        camera: Metashape camera of detection result
        f: output file.
    Returns:
        detected_img (np.ndarray): Image with detected signs annotated.
    """

    original_height, original_width = img.shape[:2]
    resized_img = cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
    results = model.predict(resized_img, conf=config.model_confidence, verbose=False, task="detect")
    result = results[0]
    boxes = result.boxes
    annotator = Annotator(img.copy())
    per_frame_detections = []
    records = []

    x_scale = original_width / WIDTH
    y_scale = original_height / HEIGHT

    for box in boxes:
        coordinates = box.xyxy[0].cpu().numpy()
        x_center = float(box.xywh[0][0].item()) * x_scale
        y_center = float(box.xywh[0][1].item()) * y_scale

        class_id = int(box.cls)
        class_name = model.names[class_id]

        x1, y1, x2, y2 = coordinates
        x1 *= x_scale
        x2 *= x_scale
        y1 *= y_scale
        y2 *= y_scale

        conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
        annotator.box_label((x1, y1, x2, y2), class_name)

        mask = np.zeros(img.shape[:2], dtype=bool)
        mask[int(max(0, y1)):int(min(img.shape[0], y2)), int(max(0, x1)):int(min(img.shape[1], x2))] = True
        per_frame_detections.append(
            {
                "frame_index": frame_index,
                "frame_name": os.path.basename(frame_path),
                "detection_id": class_id,
                "class_name": class_name,
                "x_center": float(x_center),
                "y_center": float(y_center),
                "bbox": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                "mask": mask,
                "confidence": conf,
            }
        )
        records.append(
            DetectionRecord(
                frame_index=frame_index,
                frame_name=os.path.basename(frame_path),
                frame_path=frame_path,
                detection_id=class_id,
                class_name=class_name,
                x_center=float(x_center),
                y_center=float(y_center),
                bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                mask=mask,
                confidence=conf,
                depth_m=_depth_m_for_detection(frame_path, x_center, y_center, mask),
            )
        )

    detected_img = annotator.result()

    return detected_img, len(boxes) > 0, per_frame_detections, records


def _sync_record_depth_from_frame_masks(detection_result: DetectionResult, record: DetectionRecord) -> None:
    frame_detections = detection_result.frame_masks.get(record.frame_index, [])
    target_id = str(record.detection_id)
    match = None

    for detection in frame_detections:
        if str(detection.get("detection_id", "")) != target_id:
            continue
        match = detection
        break

    if match is None:
        return

    distance_m = match.get("distance_m")
    if distance_m is None:
        return

    try:
        record.depth_m = float(distance_m)
    except (TypeError, ValueError):
        return


def write_positions_csv(detection_result: DetectionResult, workflow, csv_path: str) -> None:
    with open(csv_path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["Image", "x_px", "y_px", "X", "Y", "Z", "distance_from_camera", "type", "id", "confidence"])

        for record in detection_result.records:
            if record.depth_m is None:
                _sync_record_depth_from_frame_masks(detection_result, record)

            projection = workflow.project_detection(
                frame_index=record.frame_index,
                frame_path=record.frame_path,
                x_px=record.x_center,
                y_px=record.y_center,
                depth_m=record.depth_m,
                class_name=record.class_name,
                detection_id=record.detection_id,
                confidence=record.confidence,
            )

            if projection is None:
                continue

            record.projection = projection
            writer.writerow(
                [
                    projection.get("Image", record.frame_name),
                    f"{float(projection.get('x_px', record.x_center)):.2f}",
                    f"{float(projection.get('y_px', record.y_center)):.2f}",
                    f"{float(projection.get('X')):.6f}",
                    f"{float(projection.get('Y')):.6f}",
                    f"{float(projection.get('Z')):.6f}",
                    str(record.depth_m),
                    record.class_name,
                    record.detection_id,
                    f"{record.confidence:.4f}",
                ]
            )
            print(projection.get("Image", record.frame_name),
                                f"{float(projection.get('x_px', record.x_center)):.2f}",
                                f"{float(projection.get('y_px', record.y_center)):.2f}",
                                f"{float(projection.get('X')):.6f}",
                                f"{float(projection.get('Y')):.6f}",
                                f"{float(projection.get('Z')):.6f}",
                                str(record.depth_m),
                                record.class_name,
                                record.detection_id,
                                f"{record.confidence:.4f}",)

def detect(workflow) -> DetectionResult:
    """
    Runs sign detection on all images in the configured images directory.
    """
    global config
    detected_frame_indices = set()
    frame_masks = {}
    records = []
    if isinstance(config.frames_dir, str) and config.frames_dir.endswith(".mp4"):
        frame_paths = [config.frames_dir]
    else:
        frame_paths = glob.glob(os.path.join(config.frames_dir, "*.jpg"))
        try:
            frame_paths.sort(key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))
        except ValueError:
            frame_paths.sort()

    if (config.model == "yolo"):
        for frame_index, path in enumerate(frame_paths):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Unable to read image frame %s; skipping detection.", path)
                continue
            detected_img, has_detections, detections, frame_records = detect_animals_yolo(
                img, frame_index, path
            )
            if has_detections:
                detected_frame_indices.add(frame_index)
                frame_masks[frame_index] = detections
                records.extend(frame_records)
            cv2.imwrite(os.path.join(config.signs_detection_dir, os.path.basename(path)), detected_img)
    elif (config.model == "sam3"):
        return detect_animals_sam3(config.prompts)
        # elif (config.model == "sahi"):
        #     detect_animals_sahi()

    return DetectionResult(frame_paths, detected_frame_indices, frame_masks, records)
    
def initialize(cfg) -> None:
    """
    Initializes the detection model.

    Args:
        cfg: Configuration object with model and directory parameters.

    Returns:
        None
    """
    global model
    global config
    config = cfg
    if (config.model is None):
        raise ValueError("Signs detection model path is not specified in the configuration.")
    # if (config.model_path):
    #     model = YOLO(config.model_path)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
