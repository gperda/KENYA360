import os
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
)
import numpy as np
import warnings
import shutil
import Metashape
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
HEIGHT = 640
WIDTH = 640

model = None
config = None

def visualize_frame_output(frame_idx, video_frames, outputs):
    print(video_frames, video_frames[frame_idx])
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

def detect_animals_sahi():
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=config.model_path,
        confidence_threshold=0.4,
        device="cuda:0"
    )

    

    for image_path in glob.glob(os.path.join(config.frames_dir, "*.jpg")):
        image = read_image_as_pil(image_path)
        prediction = get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        prediction.export_visuals(config.signs_detection_dir, os.path.basename(image_path))

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

def detect_animals_sam3(chunk, f) -> np.ndarray:
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

    crs = chunk.crs
    T = chunk.transform.matrix
    cameras = chunk.cameras

    prompts = ["zebra", "elephant", "rhino", "warthog"]

    # Split frames into blocks — done once, outside all loops
    BLOCK_SIZE = 100
    start_offset = 460
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

            for local_frame_idx in range(0, len(frame_block)-1):
                frame_outputs = outputs_per_frame[local_frame_idx]
                global_frame_idx = block_start + local_frame_idx

                if frame_outputs["out_obj_ids"].shape[0] > 0:
                    detected_img = visualize_frame_output(
                        global_frame_idx,
                        video_frames_for_vis,
                        outputs=frame_outputs,
                    )
                    
                    

                if global_frame_idx > len(cameras) - 1:
                    continue
                camera = cameras[global_frame_idx + start_offset]
                if camera.transform is None:
                    continue

                for j in range(len(frame_outputs["out_obj_ids"])):
                    hi, wi = frame_outputs["out_binary_masks"].shape[1:3]
                    box_xywh = frame_outputs["out_boxes_xywh"][j]
                    prob = frame_outputs["out_probs"][j]
                    obj_id = frame_outputs["out_obj_ids"][j]
                    x, y, w, h = rel_to_abs_coords([box_xywh], wi, hi, coord_type="box")[0]
                    x_center = x + w / 2
                    y_center = y + h / 2
                    class_name = prompt

                    pt_3D = chunk.tie_points.pickPoint(
                        camera.center,
                        camera.transform.mulp(
                            camera.sensor.calibration.unproject(Metashape.Vector([x_center, y_center]))
                        ),
                    )
                    pt_3D_world = crs.project(T.mulp(pt_3D))
                    proj_xy = crs.project(T.mulp(camera.center))
                    distance = np.linalg.norm(pt_3D_world[:2] - proj_xy[:2])
                    X, Y, Z = pt_3D_world
                    line = f"{camera.label},{x_center:.2f},{y_center:.2f},{X:.6f},{Y:.6f},{Z:.6f},{distance},{class_name},{obj_id},{prob:.4f}\n"
                    f.write(line)
            shutil.rmtree(block_dir, ignore_errors=True)
            # Close session at the end of each block, still within the prompt loop
            video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

def detect_animals_yolo(img, chunk, camera, f) -> np.ndarray:
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
    
    x_scale = original_width / WIDTH
    y_scale = original_height / HEIGHT

    crs = chunk.crs
    T = chunk.transform.matrix

    for box in boxes:
        coordinates = box.xyxy[0].cpu().numpy()
        x_center = float(box.xywh[0][0].item()) * x_scale
        y_center = float(box.xywh[0][1].item()) * y_scale

        if (camera.transform == None): #Sanity check, camera not aligned
            continue
        
        pt_3D = chunk.tie_points.pickPoint(camera.center, camera.transform.mulp(camera.sensor.calibration.unproject(Metashape.Vector([x_center, y_center]))))
        pt_3D_world  = crs.project(T.mulp(pt_3D))  # transform to CRS

        proj_xy = crs.project(T.mulp(camera.center))

        distance = np.linalg.norm(pt_3D_world[:2] - proj_xy[:2])
        if distance > 25:
            continue

        class_id = int(box.cls)
        #class_name = "-".join(model.names[class_id].split("--")[:-1])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = coordinates
        x1 *= x_scale
        x2 *= x_scale
        y1 *= y_scale
        y2 *= y_scale

        annotator.box_label((x1, y1, x2, y2), class_name)
        conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
        X, Y, Z = pt_3D_world

        line = f"{camera.label}, {x_center:.2f},{y_center:.2f},{X:.6f},{Y:.6f},{Z:.6f},{distance},{class_name},{conf:.4f}\n"
        f.write(line)

    detected_img = annotator.result()

    return detected_img

def detect() -> None:
    """
    Runs sign detection on all images in the configured images directory.
    """
    global config
    doc = config.doc
    doc.open(os.path.join(config.metashape_dir, "metashape.psx"))
    chunk = doc.chunks[0]

    with open(config.signs_positions_file, "a") as f:
        f.write("Image, x_px, y_px, X, Y, Z, distance_from_camera, type, id, confidence\n")
        if (config.model == "yolo"):
            for image in os.listdir(config.frames_dir):
                if not image.endswith(".jpg"):
                    continue
                path = os.path.join(config.frames_dir, image)
                img = cv2.imread(path, cv2.IMREAD_COLOR).copy()

                camera = next((c for c in chunk.cameras if c.label == image.replace(".jpg", "")), None)
                
                detected_img= detect_animals_yolo(img, chunk, camera, f)
                cv2.imwrite(os.path.join(config.signs_detection_dir, image), detected_img)
        elif (config.model == "sam3"):
            detect_animals_sam3(chunk, f)
        elif (config.model == "sahi"):
            detect_animals_sahi()
                
    f.close()
    
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
