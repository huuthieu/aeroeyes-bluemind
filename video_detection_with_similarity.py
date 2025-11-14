"""
Video Object Detection Pipeline with Similarity Filtering

This pipeline:
1. Detects objects in video frames using YOLO-E
2. Extracts embeddings from detected objects using vision models (DINOv2)
3. Compares each detected object with a reference object
4. Keeps only objects with similarity score > threshold
"""

import cv2
import json
from pathlib import Path
from ultralytics import YOLOE
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import List, Dict, Tuple
import numpy as np


class ImageEmbeddingExtractor:
    """Extract embeddings from images using vision foundation models"""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        """
        Initialize embedding model

        Args:
            model_name: Model name from HuggingFace
            device: Device to use ('cuda', 'mps', 'cpu')
        """
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")

        # Load processor and model
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("✓ Model loaded successfully")

    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract embedding from image

        Args:
            image: PIL Image

        Returns:
            embedding: Tensor shape [1, dim]
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(self.model, 'vision_model'):
                vision_outputs = self.model.vision_model(**inputs)
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    embedding = vision_outputs.pooler_output
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    embedding = vision_outputs.last_hidden_state[:, 0, :]
                else:
                    embedding = vision_outputs.last_hidden_state.mean(dim=1)
            else:
                outputs = self.model(**inputs)

                if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                    embedding = outputs.image_embeds
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings

        Args:
            embedding1: First embedding [1, dim]
            embedding2: Second embedding [1, dim]
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            similarity: Similarity score
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
            return similarity.item()
        elif metric == "euclidean":
            distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
            similarity = 1.0 / (1.0 + distance.item())
            return similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")


class VideoDetectionWithSimilarityPipeline:
    """Video detection pipeline with similarity filtering"""

    def __init__(
        self,
        model_path: str = "yoloe-11l-seg-pf.pt",
        output_dir: str = "detection_results",
        embedding_model: str = "facebook/dinov2-base",
        similarity_threshold: float = 0.5,
        multi_ref_mode: str = "any"  # "any" or "all"
    ):
        """
        Initialize the pipeline

        Args:
            model_path: Path to YOLO-E model weights
            output_dir: Directory for outputs
            embedding_model: Vision model for embeddings
            similarity_threshold: Minimum similarity score to keep detections
            multi_ref_mode: How to combine multiple references:
                           - "any": keep object if matches ANY reference (OR logic)
                           - "all": keep object if matches ALL references (AND logic)
        """
        self.detection_model = YOLOE(model_path)
        self.embedding_extractor = ImageEmbeddingExtractor(model_name=embedding_model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.reference_embeddings = {}  # Dict to store multiple references
        self.multi_ref_mode = multi_ref_mode

        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Multi-reference mode: {multi_ref_mode}")

    def set_reference_object(self, reference_image_path: str, reference_name: str = None):
        """
        Set a single reference object for similarity comparison

        Args:
            reference_image_path: Path to reference image
            reference_name: Name/identifier for this reference (defaults to filename)
        """
        reference_path = Path(reference_image_path)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        if reference_name is None:
            reference_name = reference_path.stem

        print(f"\n📌 Loading reference object '{reference_name}': {reference_image_path}")
        reference_image = Image.open(reference_image_path).convert('RGB')
        embedding = self.embedding_extractor.extract_embedding(reference_image)
        self.reference_embeddings[reference_name] = embedding
        print(f"✓ Reference '{reference_name}' embedding extracted")

    def set_reference_objects(self, reference_paths: List[str], reference_names: List[str] = None):
        """
        Set multiple reference objects for similarity comparison

        Args:
            reference_paths: List of paths to reference images
            reference_names: List of names for references (defaults to filenames)
        """
        if reference_names is None:
            reference_names = [Path(p).stem for p in reference_paths]

        print(f"\n📌 Loading {len(reference_paths)} reference objects...")
        for ref_path, ref_name in zip(reference_paths, reference_names):
            self.set_reference_object(ref_path, ref_name)
        print(f"✓ All {len(self.reference_embeddings)} references loaded")

    def _check_similarity_match(self, crop_embedding: torch.Tensor) -> Tuple[bool, Dict]:
        """
        Check if crop embedding matches reference(s) based on multi_ref_mode.
        Uses vectorized operations for efficiency with multiple references.

        Args:
            crop_embedding: Embedding of detected crop [1, dim]

        Returns:
            Tuple of (matches: bool, similarity_scores: dict)
        """
        if not self.reference_embeddings:
            raise RuntimeError("No reference objects set. Call set_reference_object(s) first.")

        # Vectorized similarity computation
        ref_names = list(self.reference_embeddings.keys())
        ref_embeddings = torch.stack([
            self.reference_embeddings[name].squeeze(0)
            for name in ref_names
        ])  # Shape: [num_refs, dim]

        # Normalize for cosine similarity
        crop_norm = F.normalize(crop_embedding, dim=1)  # [1, dim]
        refs_norm = F.normalize(ref_embeddings, dim=1)  # [num_refs, dim]

        # Vectorized cosine similarity: [1, dim] @ [dim, num_refs] = [1, num_refs]
        similarities_tensor = torch.mm(crop_norm, refs_norm.t()).squeeze(0)  # [num_refs]

        # Convert to dict
        similarities = {
            ref_names[i]: similarities_tensor[i].item()
            for i in range(len(ref_names))
        }

        # Determine if this detection passes the filter
        if self.multi_ref_mode == "any":
            # Keep if matches ANY reference
            matches = any(sim >= self.similarity_threshold for sim in similarities.values())
        elif self.multi_ref_mode == "all":
            # Keep if matches ALL references
            matches = all(sim >= self.similarity_threshold for sim in similarities.values())
        else:
            raise ValueError(f"Unknown multi_ref_mode: {self.multi_ref_mode}")

        return matches, similarities

    def process_video(
        self,
        video_path: str,
        save_output: bool = True,
        frame_skip: int = 1,
        save_detections: bool = True,
        detection_conf: float = 0.5
    ) -> Dict:
        """
        Process video with similarity filtering

        Args:
            video_path: Path to video file
            save_output: Save output video with filtered detections
            frame_skip: Process every nth frame
            save_detections: Save detection results to JSON
            detection_conf: Confidence threshold for detection

        Returns:
            Dictionary with processing statistics
        """
        if not self.reference_embeddings:
            raise RuntimeError("Reference object(s) not set. Call set_reference_object(s) first.")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
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

        # Setup video writer
        output_video = None
        if save_output:
            output_path = self.output_dir / f"{video_path.stem}_filtered.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )

        # Detection results storage
        all_frame_detections = {}
        frame_count = 0
        total_detections = 0
        total_filtered = 0
        detections_per_frame = []

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_detections = []
                annotated_frame = frame.copy()

                if frame_count % frame_skip == 0:
                    # Run detection
                    results = self.detection_model.predict(frame, conf=detection_conf)
                    boxes = results[0].boxes

                    # Process each detection
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = results[0].names[cls_id]

                        # Crop detected region
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:  # Skip invalid crops
                            continue

                        # Convert to PIL and extract embedding
                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        crop_embedding = self.embedding_extractor.extract_embedding(crop_pil)

                        # Check similarity match with reference(s)
                        matches, similarities = self._check_similarity_match(crop_embedding)

                        total_detections += 1

                        # Filter by similarity threshold
                        if matches:
                            total_filtered += 1

                            # Get the best similarity score for display
                            best_sim = max(similarities.values())

                            detection = {
                                "class": cls_name,
                                "class_id": cls_id,
                                "confidence": conf,
                                "similarities": similarities,  # All similarity scores
                                "best_similarity": best_sim,
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

                            # Draw bbox on frame (green for passed similarity)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{cls_name} {conf:.2f} sim:{best_sim:.2f}"
                            cv2.putText(
                                annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )

                    detections_per_frame.append(len(frame_detections))

                    if output_video:
                        output_video.write(annotated_frame)
                else:
                    # Skip frame but write original
                    if output_video:
                        output_video.write(frame)
                    detections_per_frame.append(0)

                # Store detections
                all_frame_detections[str(frame_count)] = frame_detections

                frame_count += 1
                pbar.update(1)

        # Save detections
        if save_detections:
            detections_json = self.output_dir / f"{video_path.stem}_filtered_detections.json"
            with open(detections_json, "w") as f:
                json.dump(all_frame_detections, f, indent=2)

        # Cleanup
        cap.release()
        if output_video:
            output_video.release()

        # Statistics
        stats = {
            "video_path": str(video_path),
            "total_frames": total_frames,
            "frames_processed": frame_count // frame_skip if frame_skip > 1 else frame_count,
            "total_detections": total_detections,
            "filtered_detections": total_filtered,
            "filtering_ratio": total_filtered / total_detections if total_detections > 0 else 0,
            "avg_filtered_per_frame": (
                total_filtered / len(detections_per_frame)
                if detections_per_frame
                else 0
            ),
            "similarity_threshold": self.similarity_threshold,
            "output_video": str(self.output_dir / f"{video_path.stem}_filtered.mp4")
            if save_output
            else None,
            "detections_file": str(self.output_dir / f"{video_path.stem}_filtered_detections.json")
            if save_detections
            else None,
        }

        return stats


def main():
    """Main function demonstrating the combined pipeline"""

    # Configuration
    MODEL_PATH = "yoloe-11l-seg-pf.pt"
    VIDEO_PATH = "/Users/thieu.do/Desktop/learning/zaloAI/train/samples/Backpack_0/drone_video.mp4"
    OUTPUT_DIRECTORY = "detection_results"
    SIMILARITY_THRESHOLD = 0.1  # Adjust based on your needs
    EMBEDDING_MODEL = "facebook/dinov2-base"  # Options: dinov2-base, dinov2-large, dinov3-convnext-tiny-pretrain-lvd1689m
    FRAME_SKIP = 1
    DETECTION_CONF = 0.5

    # ===== OPTION 1: Single reference image =====
    # REFERENCE_IMAGES = ["train/samples/Backpack_0/object_images/img_3.jpg"]
    # MULTI_REF_MODE = "any"

    # ===== OPTION 2: Multiple reference images (keep if matches ANY) =====
    REFERENCE_IMAGES = [
        "train/samples/Backpack_0/object_images/img_3.jpg",
        "train/samples/Backpack_0/object_images/img_1.jpg",
        "train/samples/Backpack_0/object_images/img_2.jpg",
    ]
    MULTI_REF_MODE = "any"  # Keep object if matches ANY reference (OR logic)

    # ===== OPTION 3: Multiple reference images (keep if matches ALL) =====
    # REFERENCE_IMAGES = [
    #     "train/samples/Backpack_0/object_images/img_3.jpg",
    #     "train/samples/Backpack_0/object_images/img_1.jpg",
    # ]
    # MULTI_REF_MODE = "all"  # Keep object if matches ALL references (AND logic)

    # Initialize pipeline
    pipeline = VideoDetectionWithSimilarityPipeline(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIRECTORY,
        embedding_model=EMBEDDING_MODEL,
        similarity_threshold=SIMILARITY_THRESHOLD,
        multi_ref_mode=MULTI_REF_MODE
    )

    # Set reference object(s)
    try:
        if len(REFERENCE_IMAGES) == 1:
            pipeline.set_reference_object(REFERENCE_IMAGES[0])
        else:
            pipeline.set_reference_objects(REFERENCE_IMAGES)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please set valid paths to your reference object images.")
        return

    # Process video
    try:
        stats = pipeline.process_video(
            VIDEO_PATH,
            save_output=True,
            frame_skip=FRAME_SKIP,
            save_detections=True,
            detection_conf=DETECTION_CONF
        )

        print("\n" + "=" * 80)
        print("DETECTION WITH SIMILARITY FILTERING RESULTS")
        print("=" * 80)
        print(f"Video: {stats['video_path']}")
        print(f"Reference mode: {MULTI_REF_MODE}")
        print(f"Number of references: {len(REFERENCE_IMAGES)}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Filtered Detections (similarity > {stats['similarity_threshold']}): {stats['filtered_detections']}")
        print(f"Filtering Ratio: {stats['filtering_ratio']:.2%}")
        print(f"Avg Filtered/Frame: {stats['avg_filtered_per_frame']:.2f}")
        print(f"Output Video: {stats['output_video']}")
        print(f"Detections File: {stats['detections_file']}")
        print("=" * 80)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
