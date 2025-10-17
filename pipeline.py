import cv2
cv2.namedWindow("priming_window")
cv2.waitKey(1)
cv2.destroyWindow("priming_window")

import gc
import os
import subprocess
import json
from pathlib import Path
import numpy as np
import torch
import supervision as sv
from inference import get_model
from sam2.build_sam import build_sam2_camera_predictor

HOME = Path.cwd()

PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID)
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]
BASKETBALL_CLASS_ID = 0
PROXIMITY_THRESHOLD = 100

config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint_file = HOME / "segment-anything-2-real-time" / "checkpoints" / "sam2.1_hiera_large.pt"

selected_tracker_id = None

def filter_segments_by_distance(mask: np.ndarray, distance_threshold: float = 300) -> np.ndarray:
    """
    Keeps the main segment and removes segments farther than distance_threshold.

    Args:
        mask (np.ndarray): Boolean mask.
        distance_threshold (float): Maximum allowed distance from the main segment.

    Returns:
        np.ndarray: Boolean mask after filtering.
    """
    assert mask.dtype == bool, "Input mask must be boolean."
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask.copy()
    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_centroid = centroids[main_label]
    filtered_mask = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        centroid = centroids[label]
        dist = np.linalg.norm(centroid - main_centroid)
        if label == main_label or dist <= distance_threshold:
            filtered_mask[labels == label] = True
    return filtered_mask

def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert single boolean mask to RLE format."""
    from pycocotools import mask as mask_utils
    # Convert boolean mask to binary mask in Fortran order
    mask_binary = np.asfortranarray(mask.astype(np.uint8))
    # Encode the mask to RLE
    rle = mask_utils.encode(mask_binary)
    # The 'counts' object is in bytes, so we need to decode it to a string
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def save_annotations_to_json(masks_by_frame: dict, source_video_path: Path, output_folder: Path, player_name: str, motion_class: str):
    annotations_path = output_folder / f"{source_video_path.stem}-final.json"
    video_name = source_video_path.name
    
    # Get video properties from the first frame
    cap = cv2.VideoCapture(str(source_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    video_annotations = []
    for frame_idx, mask in masks_by_frame.items():
        if not np.any(mask):
            continue

        # Calculate bounding box from the mask
        y_indices, x_indices = np.where(mask)
        x_min, x_max = float(x_indices.min()), float(x_indices.max())
        y_min, y_max = float(y_indices.min()), float(y_indices.max())
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min] # COCO format is [x, y, width, height]

        frame_annotation = {
            "frame_index": frame_idx,
            "video_name": video_name,
            "fps": fps,
            "annotations": [{
                "class_name": motion_class,
                "player_name": player_name,
                "bbox": bbox,
                "segmentation": single_mask_to_rle(mask)
            }],
            "box_format": "xywh",
            "img_width": width,
            "img_height": height,
        }
        video_annotations.append(frame_annotation)

    # Save the JSON file
    with open(annotations_path, "w") as f:
        json.dump(video_annotations, f, indent=4)
    print(f"Annotations for {len(video_annotations)} frames saved to {annotations_path}")

def get_initial_detections(source_video_path_str: str):
    SOURCE_VIDEO_PATH = Path(source_video_path_str)
    if not SOURCE_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video file not found: {SOURCE_VIDEO_PATH}")
    frame_generator = sv.get_video_frames_generator(source_video_path_str)
    frame = next(frame_generator)

    result = PLAYER_DETECTION_MODEL.infer(frame, confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD)[0]
    detections = sv.Detections.from_inference(result)
    valid_class_ids = PLAYER_CLASS_IDS + [BASKETBALL_CLASS_ID]
    detections = detections[np.isin(detections.class_id, valid_class_ids)]
    return frame, detections

def process_video_and_get_masks(
    source_video_path_str: str,
    output_folder_str: str,
    selected_player_idx: int,
    first_frame: np.ndarray,
    initial_detections: sv.Detections,
    status_callback
):
    status_callback("Status: Loading segmentation model...")
    predictor = build_sam2_camera_predictor(config_file, str(checkpoint_file))

    detections = initial_detections
    global selected_tracker_id
    status_callback("Status: Initializing...")
    SOURCE_VIDEO_PATH = Path(source_video_path_str)
    OUTPUT_FOLDER = Path(output_folder_str)
    TEMP_VIDEO_PATH = OUTPUT_FOLDER / f"{SOURCE_VIDEO_PATH.stem}-mask-temp.mp4"

    TRACKE_ID = np.arange(1, len(detections.class_id) + 1)
    detections.tracker_id = TRACKE_ID

    player_detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
    if selected_player_idx >= len(player_detections.tracker_id):
        raise IndexError("Selected player index is out of bounds.")
    selected_tracker_id = player_detections.tracker_id[selected_player_idx]

    is_basketball_mask = (detections.class_id == BASKETBALL_CLASS_ID)
    ball_detections = detections[is_basketball_mask]
    ball_tracker_ids = ball_detections.tracker_id
    ball_tracker_ids_set = set(ball_tracker_ids)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.load_first_frame(first_frame)

        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            xyxy = np.array([xyxy])

            _, object_ids, mask_logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=tracker_id,
                bbox=xyxy
            )

    all_masks = {}
    cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        status_callback(f"Status: Processing frame {index + 1}/{total_frames}")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = predictor.track(frame)
            tracker_ids = np.array(tracker_ids)
            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)

            masks = np.array([
                filter_segments_by_distance(mask, distance_threshold=300)
                for mask
                in masks
            ])

            # Create detections for ALL players (SAM2 tracks all)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                tracker_id=tracker_ids
            )

            # Filter ONLY the selected player or the basketball for visualization
            tracker_ids_to_show = [selected_tracker_id] + [tid for tid in tracker_ids if tid in ball_tracker_ids_set]
            detections_to_show = detections[np.isin(detections.tracker_id, tracker_ids_to_show)]

            output_frame = np.zeros_like(frame)
            selected_player_mask_idx = np.where(detections_to_show.tracker_id == selected_tracker_id)[0]

            if len(selected_player_mask_idx) > 0:
                player_mask = detections_to_show.mask[selected_player_mask_idx[0]]
                output_frame[player_mask] = (255, 255, 255)
                
                # Add ball mask(s)
                for i, tracker_id in enumerate(detections_to_show.tracker_id):
                    if tracker_id in ball_tracker_ids_set:
                        ball_mask = detections_to_show.mask[i]
                        output_frame[ball_mask] = (255, 255, 255)
                
                # Keep only largest connected component
                output_gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_gray, connectivity=8)
                
                if num_labels > 2:  # Multiple white regions exist
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    output_frame = np.zeros_like(frame)
                    output_frame[labels == largest_label] = (255, 255, 255)
                    all_masks[index] = (labels == largest_label)  # This creates a 2D boolean array
                else:  # Only one region (player + ball still connected)
                    # Get combined mask from output_frame for consistency
                    combined_mask = (output_gray > 0)  # Convert back to boolean
                    all_masks[index] = combined_mask
            else:
                all_masks[index] = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)

            return output_frame

    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TEMP_VIDEO_PATH,
        callback=callback,
        show_progress=False
    )

    del predictor
    torch.cuda.empty_cache()
    gc.collect()

    status_callback("Status: Preview ready. Click 'Confirm & Save' to finish.")
    return all_masks, str(TEMP_VIDEO_PATH)

def finalize_and_save(
    temp_video_path_str: str,
    all_masks: dict,
    output_folder_str: str,
    source_video_path_str: str,
    player_name: str,
    motion_class: str,
    status_callback
):
    """Compresses the video, saves JSON, and cleans up the temporary file."""
    try:
        status_callback("Status: Compressing and saving final files...")
        TEMP_VIDEO_PATH = Path(temp_video_path_str)
        OUTPUT_FOLDER = Path(output_folder_str)
        SOURCE_VIDEO_PATH = Path(source_video_path_str)
        FINAL_VIDEO_PATH = OUTPUT_FOLDER / f"{SOURCE_VIDEO_PATH.stem}-final.mp4"
        
        ffmpeg_command = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(TEMP_VIDEO_PATH),
            '-vcodec', 'mpeg4',   # <-- Use the universal mpeg4 codec
            '-q:v', '4',         # <-- Use a quality scale (lower is better quality)
            str(FINAL_VIDEO_PATH)
        ]
        subprocess.run(ffmpeg_command, check=True)
        status_callback("Status: Saving annotations...")
        save_annotations_to_json(all_masks, FINAL_VIDEO_PATH, OUTPUT_FOLDER, player_name, motion_class)
        os.remove(TEMP_VIDEO_PATH)
        status_callback("Status: Done!")
        return str(FINAL_VIDEO_PATH)
    except Exception as e:
        status_callback(f"ERROR during save: {e}")
        return None