"""
Process public_test dataset using mask-based detection and generate submission JSON

This script combines mask extraction (BEN2) with YOLO detection and DINOv3 similarity matching.
It processes all videos in public_test/samples using masked crops for more accurate detection.

Workflow:
1. Extract background masks from reference images using BEN2 model
2. Extract Visual Prompt Embeddings (VPE) from masked reference images
3. Pre-compute DINOv3 embeddings for reference object images
4. For each video frame:
   - Detect objects with YOLO, extract masked crops
   - Compute DINOv3 embeddings for crops
   - Find best matching reference image for each crop
   - Filter by similarity threshold
5. Generate submission JSON with detections
"""

import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from itertools import batched
from tqdm import tqdm
from ultralytics.models.yolo import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from torchvision.ops import masks_to_boxes, roi_align
from torchvision.transforms.functional import to_pil_image

# ========== CONFIGURATION ==========
MODEL_DIR = Path("models")
BATCH_SIZE = 1

# ORT Providers configuration
ORT_PROVIDERS = [
    "CUDAExecutionProvider",
    (
        "CoreMLExecutionProvider",
        {
            "ModelFormat": "MLProgram",
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "1",
        },
    ),
    "CPUExecutionProvider",
]

# Type mapping for ONNX Runtime
ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(uint16)": np.uint16,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(double)": np.float64,
    "tensor(bool)": bool,
    "tensor(float16)": np.float16,
}


# ========== HELPER FUNCTIONS ==========
def get_ort_session_device_type(session: ort.InferenceSession) -> str:
    """Get device type from ONNX Runtime session."""
    provider = session.get_providers()[0]
    return provider[: provider.index("ExecutionProvider")].lower()


# ========== STEP 1: INITIALIZE BEN2 MODEL ==========
def setup_ben2_session():
    """Initialize BEN2 ONNX session for background extraction."""
    session = ort.InferenceSession(
        MODEL_DIR / "BEN2-folded.onnx", providers=ORT_PROVIDERS
    )
    io_binding = session.io_binding()
    input_node = session.get_inputs()[0]
    output_node = session.get_outputs()[0]

    device_type = get_ort_session_device_type(session)
    if device_type == "coreml":
        device_type = "cpu"

    b, c, h, w = input_node.shape
    input_batch = np.empty([b, c, h, w], dtype=np.float32)

    input_ortvalue = None
    if device_type != "cpu":
        input_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
            input_node.shape, ORT_TYPE_TO_NUMPY[input_node.type], device_type
        )
        io_binding.bind_ortvalue_input(input_node.name, input_ortvalue)
    else:
        io_binding.bind_cpu_input(input_node.name, input_batch)

    output_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
        output_node.shape, ORT_TYPE_TO_NUMPY[output_node.type], device_type
    )
    io_binding.bind_ortvalue_output(output_node.name, output_ortvalue)

    return session, io_binding, input_batch, input_ortvalue, output_ortvalue, device_type


# ========== STEP 2: PROCESS REF IMAGES WITH BEN2 ==========
def process_ref_images_ben2(ref_paths):
    """Extract masks from reference images using BEN2 model."""
    session, io_binding, input_batch, input_ortvalue, output_ortvalue, device_type = (
        setup_ben2_session()
    )

    input_rgb_list: list[np.ndarray | None] = [None] * BATCH_SIZE

    for batch_paths in batched(tqdm(ref_paths, desc="Processing ref images with BEN2"), BATCH_SIZE):
        for idx, img_path in enumerate(batch_paths):
            input_rgb_list[idx] = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if input_rgb_list[idx] is None:
                print(f"Warning: Could not read {img_path}")
                continue

            img_data = input_rgb_list[idx]
            resized_image = cv2.resize(img_data, (1024, 1024))
            chw_image = resized_image.transpose(2, 0, 1)
            np.divide(chw_image, np.iinfo(np.uint8).max, out=input_batch[idx])

        # Inference
        if device_type != "cpu" and input_ortvalue is not None:
            input_ortvalue.update_inplace(input_batch)
        session.run_with_iobinding(io_binding)
        outputs = output_ortvalue.numpy()

        # Postprocess
        for img_path, input_rgb, output in zip(batch_paths, input_rgb_list, outputs):
            if input_rgb is None:
                continue

            raw_mask = output.squeeze()
            min_val = raw_mask.min()
            max_val = raw_mask.max()

            normalized_mask = (raw_mask - min_val) / (
                max_val - min_val + np.finfo(np.float32).eps
            )
            normalized_mask *= np.iinfo(np.uint8).max

            resized_mask = cv2.resize(
                normalized_mask.astype(np.uint8),
                dsize=(input_rgb.shape[1], input_rgb.shape[0]),
            )

            # Save background mask
            save_path = img_path.with_name(f"{img_path.stem}_bg.png")
            cv2.imwrite(str(save_path), resized_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Save object with alpha channel
            bgra_image = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2BGRA)
            bgra_image[:, :, 3] = resized_mask
            save_path = img_path.with_name(f"{img_path.stem}_obj.png")
            cv2.imwrite(str(save_path), bgra_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


# ========== STEP 3: EXTRACT VPE FROM REF IMAGES ==========
def extract_vpe_embeddings(ref_paths, yoloe):
    """Extract visual prompt embeddings from masked reference images."""
    predictor = YOLOEVPDetectPredictor(
        overrides={"task": "detect", "mode": "predict", "batch": 1}
    )
    predictor.setup_model(yoloe.model)

    all_vpe = []

    for ref_path in tqdm(ref_paths, desc="Extracting VPE from masks"):
        bg_path = ref_path.with_name(f"{ref_path.stem}_bg.png")
        bg_np = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
        if bg_np is None:
            print(f"Warning: Could not read {bg_path}")
            continue

        # Normalize mask to binary (0 or 1)
        bg_mask = (bg_np > (255 // 2)).astype(np.uint8)

        predictor.set_prompts(
            {
                "masks": [bg_mask],  # Keep as numpy array, not tensor
                "cls": np.array([0]),
            }
        )

        vpe = predictor.get_vpe(ref_path)
        all_vpe.append(vpe)

    if not all_vpe:
        print("Warning: No VPE embeddings extracted!")
        return None

    avg_vpe = torch.mean(torch.stack(all_vpe), dim=0)
    return avg_vpe


# ========== STEP 4: PROCESS VIDEO FRAMES WITH MASKED DETECTION ==========
def extract_masked_crops(yoloe, frame):
    """Detect objects in frame and extract masked crops using YOLO.

    Args:
        yoloe: YOLO model
        frame: numpy array of frame image

    Returns:
        crops: Tensor of detected object crops [N, 3, 224, 224]
        boxes: Bounding boxes of detections [N, 4]
        or (None, None) if no detections
    """
    # Run YOLO detection
    results = yoloe.predict(frame, conf=0.0001, iou=0.01)
    if not results or results[0].masks is None:
        return None, None

    result = results[0]

    # Convert original image to tensor
    img = torch.as_tensor(result.orig_img)
    img = img.permute(2, 0, 1)
    img = img.float() / 255.0
    H_img, W_img = img.shape[-2:]

    # Get masks from detection
    masks = torch.as_tensor(result.masks.data).float()
    N, _, _ = masks.shape

    # Resize masks to match image size
    masks = F.interpolate(
        masks.unsqueeze(1),
        size=(H_img, W_img),
        mode="nearest",
    ).squeeze(1)

    # Get bounding boxes from masks
    boxes = masks_to_boxes(masks)

    # Create masked images
    img_batch = img.unsqueeze(0).expand(N, -1, -1, -1)
    masks_batched = masks.unsqueeze(1)
    masked_imgs = img_batch * masks_batched

    # Prepare ROIs for roi_align
    batch_idx = torch.arange(N).float().unsqueeze(1)
    rois = torch.cat([batch_idx, boxes], dim=1)

    # Extract crops using roi_align
    crops = roi_align(
        masked_imgs,
        rois,
        output_size=(224, 224),
        spatial_scale=1.0,
        aligned=True,
    )

    return crops, boxes.numpy()


# ========== STEP 5: COMPUTE DINO EMBEDDINGS AND SIMILARITY ==========
def setup_dino_model():
    """Initialize and cache DINOv3 model and processor."""
    dinov3_path = MODEL_DIR / "dinov3"

    if not dinov3_path.exists():
        print(f"   Warning: DINOv3 model not found at {dinov3_path}")
        return None, None

    model = AutoModel.from_pretrained(
        str(dinov3_path),
        device_map="auto",
    )
    processor = AutoImageProcessor.from_pretrained(str(dinov3_path))

    return model, processor


def compute_ref_embeddings(ref_paths, model, processor):
    """Pre-compute DINOv3 embeddings for reference object images.

    Args:
        ref_paths: List of reference image paths
        model: DINOv3 model
        processor: Image processor

    Returns:
        ref_embeddings: Tensor of reference embeddings [N_refs, hidden_dim]
    """
    if model is None or processor is None:
        return None

    # Process reference object images
    ref_obj_paths = [path.with_name(f"{path.stem}_obj.png") for path in ref_paths]
    ref_obj_paths = [p for p in ref_obj_paths if p.exists()]

    if not ref_obj_paths:
        print(f"   Warning: No reference object images found")
        return None

    # Load images safely
    ref_images = []
    for path in ref_obj_paths:
        try:
            img = load_image(str(path))
            if img is not None:
                ref_images.append(img)
        except Exception as e:
            print(f"   Warning: Could not load reference image {path}: {e}")
            continue

    if not ref_images:
        print(f"   Warning: Could not load any reference object images")
        return None

    try:
        ref_pixels = processor(
            images=ref_images,
            return_tensors="pt",
            device=model.device,
        )
    except Exception as e:
        print(f"   Warning: Error processing reference images: {e}")
        return None

    with torch.inference_mode():
        ref_input = model(**ref_pixels)
        ref_embeddings = ref_input.pooler_output

    return ref_embeddings


def compute_dino_similarity(crops, model, processor, ref_embeddings):
    """Compute DINOv3 embeddings and cosine similarity with cached reference embeddings.

    Args:
        crops: Tensor of detected object crops [N, 3, 224, 224]
        model: DINOv3 model
        processor: Image processor
        ref_embeddings: Pre-computed reference embeddings [N_refs, hidden_dim]

    Returns:
        similarities: Tensor of similarity scores [N, N_refs] - similarities for each crop with each reference
    """
    if crops is None or model is None or ref_embeddings is None:
        return None

    # Process detected crops
    try:
        dino_inputs = []
        for crop in crops:
            try:
                pil_img = to_pil_image(crop[[2, 1, 0], ...])
                img = load_image(pil_img)
                if img is not None:
                    dino_inputs.append(img)
            except Exception as e:
                continue

        if not dino_inputs:
            return None

        dino_pixels = processor(
            images=dino_inputs,
            return_tensors="pt",
            device=model.device,
        )
    except Exception as e:
        print(f"   Warning: Error processing crop images: {e}")
        return None

    with torch.inference_mode():
        dino_input = model(**dino_pixels)
        dino_embeddings = dino_input.pooler_output

        # Compute cosine similarity matrix [N_crops, N_refs]
        # Each row is similarities of one crop to all references
        M = F.cosine_similarity(
            dino_embeddings.unsqueeze(1),  # [N_crops, 1, hidden_dim]
            ref_embeddings.unsqueeze(0),   # [1, N_refs, hidden_dim]
            dim=-1,
        )
        # M shape: [N_crops, N_refs]
        # Caller will find best match using argmax

        return M


def group_detections_by_intervals(
    detections_per_frame: Dict[int, List[Dict]]
) -> List[List[Dict]]:
    """
    Group consecutive frame detections into intervals

    Args:
        detections_per_frame: Dict mapping frame number to list of detections

    Returns:
        List of intervals, each interval is a list of bbox dicts
    """
    if not detections_per_frame:
        return []

    # Sort frames
    sorted_frames = sorted(detections_per_frame.keys())

    intervals = []
    current_interval = []
    last_frame = None

    for frame in sorted_frames:
        detections = detections_per_frame[frame]

        if not detections:
            # No detections in this frame
            if current_interval:
                intervals.append(current_interval)
                current_interval = []
            last_frame = None
            continue

        # Check if this frame is consecutive with previous
        if last_frame is not None and frame - last_frame > 1:
            # Gap detected, start new interval
            if current_interval:
                intervals.append(current_interval)
            current_interval = []

        # Add all detections from this frame
        for det in detections:
            bbox = det.get("bbox", {})
            current_interval.append({
                "frame": frame,
                "x1": int(bbox.get("x1", 0)),
                "y1": int(bbox.get("y1", 0)),
                "x2": int(bbox.get("x2", 0)),
                "y2": int(bbox.get("y2", 0))
            })

        last_frame = frame

    # Add last interval if exists
    if current_interval:
        intervals.append(current_interval)

    return intervals


def process_video_for_submission_mask(
    yoloe,
    video_path: Path,
    ref_paths: List[Path],
    dino_model,
    dino_processor,
    ref_embeddings,
    frame_skip: int = 1,
    detection_conf: float = 0.5,
    similarity_threshold: float = 0.0
) -> Dict[int, List[Dict]]:
    """
    Process a single video using masked detection and return detections in submission format.

    For each frame, keeps only the crop with highest similarity to any reference image.
    Filters by similarity threshold.

    Args:
        yoloe: YOLO model
        video_path: Path to video file
        ref_paths: List of reference image paths
        dino_model: Pre-loaded DINOv3 model
        dino_processor: Pre-loaded image processor
        ref_embeddings: Pre-computed reference embeddings
        frame_skip: Process every nth frame
        detection_conf: Confidence threshold for detection
        similarity_threshold: Minimum similarity score to keep detection

    Returns:
        Dict mapping frame number to list of detections
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Detection results storage
    detections_per_frame = {}
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_detections = []

            if frame_count % frame_skip == 0:
                # Run masked detection
                crops, boxes = extract_masked_crops(yoloe, frame)

                if crops is not None and boxes is not None:
                    # Compute similarity with DINOv3 using cached models and embeddings
                    similarities = compute_dino_similarity(
                        crops, dino_model, dino_processor, ref_embeddings
                    )

                    if similarities is not None:
                        # Find the crop with the highest similarity score in this frame
                        # similarities shape: [N_crops, N_refs]
                        # For each crop, get max similarity across all refs
                        max_sims_per_crop = similarities.max(dim=1)[0]  # [N_crops]
                        best_crop_idx = max_sims_per_crop.argmax().item()  # Index of crop with highest similarity
                        best_crop_score = float(max_sims_per_crop[best_crop_idx].cpu() if isinstance(max_sims_per_crop[best_crop_idx], torch.Tensor) else max_sims_per_crop[best_crop_idx])

                        # Filter by similarity threshold
                        if best_crop_score >= similarity_threshold:
                            # Get best matching reference for this best crop
                            best_crop_sims = similarities[best_crop_idx]  # [N_refs]
                            best_ref_idx = int(best_crop_sims.argmax().cpu() if isinstance(best_crop_sims.argmax(), torch.Tensor) else best_crop_sims.argmax())

                            box = boxes[best_crop_idx]
                            x1, y1, x2, y2 = box
                            best_ref_name = ref_paths[best_ref_idx].name if best_ref_idx < len(ref_paths) else "unknown"

                            # Create detection entry for the best crop only
                            detection = {
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2),
                                },
                                "best_match_idx": best_ref_idx,
                                "best_match_ref": best_ref_name,
                                "similarity": best_crop_score,
                            }
                            frame_detections.append(detection)

            # Store detections (even if empty)
            detections_per_frame[frame_count] = frame_detections

            frame_count += 1
            pbar.update(1)

    cap.release()
    return detections_per_frame


def format_submission_entry(
    video_id: str,
    detections_per_frame: Dict[int, List[Dict]]
) -> Dict[str, Any]:
    """
    Format detections for a single video into submission format

    Args:
        video_id: Video identifier
        detections_per_frame: Dict mapping frame number to list of detections

    Returns:
        Formatted entry for submission JSON
    """
    # Group detections into intervals
    intervals = group_detections_by_intervals(detections_per_frame)

    # Format intervals
    formatted_detections = []
    for interval in intervals:
        if interval:  # Only add non-empty intervals
            formatted_detections.append({
                "bboxes": interval
            })

    return {
        "video_id": video_id,
        "detections": formatted_detections if formatted_detections else []
    }


def process_public_test_mask(
    samples_dir: Path,
    output_file: Path,
    model_path: str = "models/yoloe-v8l-seg.pt",
    frame_skip: int = 1,
    detection_conf: float = 0.5,
    similarity_threshold: float = 0.0,
) -> None:
    """
    Process all videos in public_test using mask-based detection and generate submission JSON

    Args:
        samples_dir: Directory containing video samples
        output_file: Path to output submission JSON file
        model_path: Path to YOLO model
        frame_skip: Process every nth frame
        detection_conf: Confidence threshold for detection
        similarity_threshold: Minimum similarity score to keep detection
    """
    print("=" * 80)
    print("PROCESSING PUBLIC_TEST DATASET WITH MASK-BASED DETECTION")
    print("=" * 80)

    # Initialize YOLO model
    print("\nInitializing YOLO model...")
    yoloe = YOLOE(model_path).eval()

    # Initialize DINOv3 model (global for all videos)
    print("Initializing DINOv3 model...")
    dino_model, dino_processor = setup_dino_model()
    if dino_model is None:
        print("Warning: DINOv3 model could not be loaded")

    # Get all video directories
    all_video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])

    if not all_video_dirs:
        print(f"Error: No video directories found in {samples_dir}")
        return

    print(f"Total videos found: {len(all_video_dirs)}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    submission = []

    for video_idx, video_dir in enumerate(all_video_dirs):
        video_id = video_dir.name
        video_path = video_dir / "drone_video.mp4"
        object_images_dir = video_dir / "object_images"

        if not video_path.exists():
            print(f"\n⚠️  Video not found: {video_path}")
            submission.append({
                "video_id": video_id,
                "detections": []
            })
            continue

        # Get reference images
        reference_images = sorted(object_images_dir.glob("*.jpg"))
        if not reference_images:
            print(f"\n⚠️  No reference images found for {video_id}")
            submission.append({
                "video_id": video_id,
                "detections": []
            })
            continue

        print(f"\n📹 Processing [{video_idx + 1}/{len(all_video_dirs)}]: {video_id}")
        print(f"   Reference images: {len(reference_images)}")

        try:
            # Step 1: Process reference images with BEN2
            print("   - Extracting masks from reference images...")
            process_ref_images_ben2(reference_images)

            # Step 2: Extract VPE embeddings
            print("   - Extracting VPE embeddings...")
            avg_vpe = extract_vpe_embeddings(reference_images, yoloe)
            if avg_vpe is not None:
                yoloe.set_classes(["obj"], avg_vpe)

            # Step 3: Pre-compute DINOv3 reference embeddings
            print("   - Pre-computing DINOv3 reference embeddings...")
            ref_embeddings = compute_ref_embeddings(
                reference_images, dino_model, dino_processor
            )

            # Step 4: Process video with masked detection
            print("   - Processing video frames with masked detection...")
            detections_per_frame = process_video_for_submission_mask(
                yoloe=yoloe,
                video_path=video_path,
                ref_paths=reference_images,
                dino_model=dino_model,
                dino_processor=dino_processor,
                ref_embeddings=ref_embeddings,
                frame_skip=frame_skip,
                detection_conf=detection_conf,
                similarity_threshold=similarity_threshold,
            )

            # Step 5: Format entry
            entry = format_submission_entry(video_id, detections_per_frame)
            submission.append(entry)

            # Print stats
            total_detections = sum(len(dets) for dets in detections_per_frame.values())
            intervals = len(entry["detections"])
            print(f"   ✓ Detections: {total_detections} in {intervals} interval(s)")

        except Exception as e:
            print(f"   ✗ Error processing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            submission.append({
                "video_id": video_id,
                "detections": []
            })

    # Write submission file
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)

    print("\n" + "=" * 80)
    print("SUBMISSION GENERATED")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Total videos: {len(submission)}")
    print(f"Videos with detections: {sum(1 for e in submission if e['detections'])}")
    print(f"Videos without detections: {sum(1 for e in submission if not e['detections'])}")
    print("=" * 80)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process public_test dataset using mask-based detection and generate submission JSON"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="public_test/samples",
        help="Directory containing video samples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_public_test_mask.json",
        help="Output submission JSON file path"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/yoloe-v8l-seg.pt",
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every nth frame"
    )
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score to keep detection (0.0 to 1.0)"
    )

    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_file = Path(args.output)

    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        return

    process_public_test_mask(
        samples_dir=samples_dir,
        output_file=output_file,
        model_path=args.model_path,
        frame_skip=args.frame_skip,
        detection_conf=args.detection_conf,
        similarity_threshold=args.similarity_threshold
    )


if __name__ == "__main__":
    main()
