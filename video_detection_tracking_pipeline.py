"""
Video Object Detection and Tracking Pipeline using YOLO-E

This pipeline processes video files with object detection and tracking,
saving both labels, bounding boxes, and tracking IDs for each frame.
"""

import cv2
import json
from pathlib import Path
from ultralytics import YOLOE
from tqdm import tqdm


class VideoDetectionTrackingPipeline:
    def __init__(self, model_path="yoloe-11l-seg-pf.pt", output_dir="detection_results"):
        """
        Initialize the video detection and tracking pipeline.

        Args:
            model_path: Path to the YOLO-E model weights
            output_dir: Directory to save detection and tracking results
        """
        self.model = YOLOE(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_video(self, video_path, save_output=True, frame_skip=1, save_detections=True):
        """
        Process a single video file for object detection and tracking.

        Args:
            video_path: Path to the video file
            save_output: Whether to save the output video with detections and tracks
            frame_skip: Process every nth frame (1 = all frames, 2 = every 2nd frame, etc.)
            save_detections: Whether to save detection labels, boxes, and track IDs as JSON

        Returns:
            Dictionary containing detection and tracking statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video capture
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing: {video_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

        # Setup video writer if saving output
        output_video = None
        if save_output:
            output_path = self.output_dir / f"{video_path.stem}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )

        # Create detections JSON file
        detections_json = None
        all_frame_detections = {}
        if save_detections:
            detections_json = self.output_dir / f"{video_path.stem}_detections_tracked.json"

        # Track statistics
        track_ids_seen = set()
        frame_count = 0
        detection_count = 0
        detections_per_frame = []

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame based on frame_skip
                if frame_count % frame_skip == 0:
                    # Run detection with tracking
                    results = self.model.track(frame, conf=0.5, persist=True)

                    # Draw bounding boxes and track IDs on frame
                    annotated_frame = results[0].plot()

                    # Extract detection and tracking data
                    boxes = results[0].boxes
                    frame_detections = []

                    for box in boxes:
                        # Get bounding box coordinates
                        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = results[0].names[cls_id]

                        # Get tracking ID
                        track_id = int(box.id[0]) if box.id is not None else None
                        if track_id is not None:
                            track_ids_seen.add(track_id)

                        detection = {
                            "class": cls_name,
                            "class_id": cls_id,
                            "track_id": track_id,
                            "confidence": conf,
                            "bbox": {
                                "x1": xyxy[0],
                                "y1": xyxy[1],
                                "x2": xyxy[2],
                                "y2": xyxy[3],
                                "width": xyxy[2] - xyxy[0],
                                "height": xyxy[3] - xyxy[1],
                            },
                        }
                        frame_detections.append(detection)

                    # Store frame detections
                    all_frame_detections[str(frame_count)] = frame_detections

                    # Count detections
                    num_detections = len(frame_detections)
                    detection_count += num_detections
                    detections_per_frame.append(num_detections)

                    if output_video:
                        output_video.write(annotated_frame)
                else:
                    # Skip processing, just write original frame
                    if output_video:
                        output_video.write(frame)
                    # Store empty detections for skipped frames
                    if save_detections:
                        all_frame_detections[str(frame_count)] = []

                frame_count += 1
                pbar.update(1)

        # Save detections to JSON file
        if save_detections and detections_json:
            with open(detections_json, "w") as f:
                json.dump(all_frame_detections, f, indent=2)

        # Cleanup
        cap.release()
        if output_video:
            output_video.release()

        # Calculate statistics
        stats = {
            "video_path": str(video_path),
            "total_frames": total_frames,
            "frames_processed": frame_count // frame_skip if frame_skip > 1 else frame_count,
            "total_detections": detection_count,
            "unique_tracks": len(track_ids_seen),
            "avg_detections_per_frame": (
                detection_count / len(detections_per_frame)
                if detections_per_frame
                else 0
            ),
            "output_video": str(self.output_dir / f"{video_path.stem}_tracked.mp4")
            if save_output
            else None,
            "detections_file": str(detections_json) if save_detections else None,
        }

        return stats


def main():
    """Main entry point for the video detection and tracking pipeline."""

    # Configuration
    MODEL_PATH = "yoloe-11l-seg-pf.pt"
    VIDEO_PATH = "/Users/thieu.do/Desktop/learning/zaloAI/train/samples/Backpack_0/drone_video.mp4"
    OUTPUT_DIRECTORY = "detection_results"
    FRAME_SKIP = 1  # Process every frame (set to 2+ to skip frames for speed)

    # Initialize pipeline
    pipeline = VideoDetectionTrackingPipeline(
        model_path=MODEL_PATH, output_dir=OUTPUT_DIRECTORY
    )

    # Process video
    try:
        stats = pipeline.process_video(VIDEO_PATH, save_output=True, frame_skip=FRAME_SKIP, save_detections=True)
        print("\n" + "=" * 80)
        print("DETECTION AND TRACKING RESULTS")
        print("=" * 80)
        print(f"Video: {stats['video_path']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Unique Tracks: {stats['unique_tracks']}")
        print(f"Avg Detections/Frame: {stats['avg_detections_per_frame']:.2f}")
        print(f"Output Video: {stats['output_video']}")
        print(f"Detections File: {stats['detections_file']}")
        print("=" * 80)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
