"""
Process public_test dataset and generate submission JSON

This script:
1. Processes all videos in public_test/samples
2. Uses object_images from each video as reference
3. Generates detections with similarity filtering
4. Outputs submission JSON in the required format
"""

import json
import cv2
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from video_detection_with_similarity import VideoDetectionWithSimilarityPipeline


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


def process_video_for_submission(
    pipeline: VideoDetectionWithSimilarityPipeline,
    video_path: Path,
    reference_images: List[Path],
    frame_skip: int = 1,
    detection_conf: float = 0.5
) -> Dict[int, List[Dict]]:
    """
    Process a single video and return detections in submission format
    
    Args:
        pipeline: VideoDetectionWithSimilarityPipeline instance
        video_path: Path to video file
        reference_images: List of reference image paths
        frame_skip: Process every nth frame
        detection_conf: Confidence threshold for detection
        
    Returns:
        Dict mapping frame number to list of detections
    """
    import cv2
    from PIL import Image
    import torch.nn.functional as F
    
    # Set reference objects
    if len(reference_images) == 1:
        pipeline.set_reference_object(str(reference_images[0]))
    else:
        pipeline.set_reference_objects([str(img) for img in reference_images])
    
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
                # Run detection
                results = pipeline.detection_model.predict(frame, conf=detection_conf)
                boxes = results[0].boxes
                
                # Process each detection
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Crop detected region
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:  # Skip invalid crops
                        continue
                    
                    # Convert to PIL and extract embedding
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    crop_embedding = pipeline.embedding_extractor.extract_embedding(crop_pil)
                    
                    # Check similarity match with reference(s)
                    matches, similarities = pipeline._check_similarity_match(crop_embedding)
                    
                    # Filter by similarity threshold
                    if matches:
                        detection = {
                            "bbox": {
                                "x1": xyxy[0],
                                "y1": xyxy[1],
                                "x2": xyxy[2],
                                "y2": xyxy[3],
                            },
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


def process_public_test(
    samples_dir: Path,
    output_file: Path,
    model_path: str = "yoloe-11l-seg-pf.pt",
    embedding_model: str = "facebook/dinov2-base",
    similarity_threshold: float = 0.1,
    frame_skip: int = 1,
    detection_conf: float = 0.5,
    multi_ref_mode: str = "any",
    save_output_videos: bool = False
) -> None:
    """
    Process all videos in public_test and generate submission JSON
    
    Args:
        samples_dir: Directory containing video samples
        output_file: Path to output submission JSON file
        model_path: Path to YOLO model
        embedding_model: Embedding model name
        similarity_threshold: Similarity threshold for filtering
        frame_skip: Process every nth frame
        detection_conf: Confidence threshold for detection
        multi_ref_mode: "any" or "all" for multiple reference images
        save_output_videos: Whether to save output videos
    """
    # Initialize pipeline
    output_dir = Path("public_test_results")
    output_dir.mkdir(exist_ok=True)
    
    pipeline = VideoDetectionWithSimilarityPipeline(
        model_path=model_path,
        output_dir=output_dir,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        multi_ref_mode=multi_ref_mode
    )
    
    # Get all video directories
    all_video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    video_dirs = all_video_dirs if all_video_dirs else []
    
    print("=" * 80)
    print("PROCESSING PUBLIC_TEST DATASET")
    print("=" * 80)
    print(f"Total videos found: {len(all_video_dirs)}")
    print(f"Processing first video only: {video_dirs[0].name if video_dirs else 'None'}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    submission = []
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        video_path = video_dir / "drone_video.mp4"
        object_images_dir = video_dir / "object_images"
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            submission.append({
                "video_id": video_id,
                "detections": []
            })
            continue
        
        # Get reference images
        reference_images = sorted(object_images_dir.glob("*.jpg"))
        if not reference_images:
            print(f"⚠️  No reference images found for {video_id}")
            submission.append({
                "video_id": video_id,
                "detections": []
            })
            continue
        
        print(f"\n📹 Processing: {video_id}")
        print(f"   Reference images: {len(reference_images)}")
        
        try:
            # Process video
            detections_per_frame = process_video_for_submission(
                pipeline=pipeline,
                video_path=video_path,
                reference_images=reference_images,
                frame_skip=frame_skip,
                detection_conf=detection_conf
            )
            
            # Format entry
            entry = format_submission_entry(video_id, detections_per_frame)
            submission.append(entry)
            
            # Print stats
            total_detections = sum(len(dets) for dets in detections_per_frame.values())
            intervals = len(entry["detections"])
            print(f"   ✓ Detections: {total_detections} in {intervals} interval(s)")
            
        except Exception as e:
            print(f"   ✗ Error processing {video_id}: {e}")
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
        description="Process public_test dataset and generate submission JSON"
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
        default="submission_public_test.json",
        help="Output submission JSON file path"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="yoloe-11l-seg-pf.pt",
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="facebook/dinov2-base",
        help="Embedding model name"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.1,
        help="Similarity threshold for filtering"
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
        "--multi-ref-mode",
        type=str,
        choices=["any", "all"],
        default="any",
        help="Mode for multiple reference images: 'any' (OR) or 'all' (AND)"
    )
    
    args = parser.parse_args()
    
    samples_dir = Path(args.samples_dir)
    output_file = Path(args.output)
    
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        return
    
    process_public_test(
        samples_dir=samples_dir,
        output_file=output_file,
        model_path=args.model_path,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        frame_skip=args.frame_skip,
        detection_conf=args.detection_conf,
        multi_ref_mode=args.multi_ref_mode
    )


if __name__ == "__main__":
    main()

