"""
Script để đánh giá độ chính xác sử dụng ST-IoU (Spatio-Temporal IoU)
So sánh kết quả dự đoán với ground truth annotations

ST-IoU xem xét cả không gian (bounding box) và thời gian (frame) như một khối 3D
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
from tqdm import tqdm


def calculate_spatial_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Tính Spatial IoU giữa 2 bounding boxes

    Args:
        bbox1, bbox2: (x1, y1, x2, y2) coordinates

    Returns:
        IoU score (0 to 1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Tính diện tích giao
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Tính diện tích hợp
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_temporal_iou(frames1: Set[int], frames2: Set[int]) -> float:
    """
    Tính Temporal IoU giữa 2 tập hợp frames

    Args:
        frames1, frames2: Tập hợp frame IDs

    Returns:
        Temporal IoU score (0 to 1)
    """
    intersection = len(frames1 & frames2)
    union = len(frames1 | frames2)

    if union == 0:
        return 0.0

    return intersection / union


def calculate_st_iou(
    pred_bboxes: List[Dict],
    gt_bboxes: List[Dict]
) -> float:
    """
    Tính ST-IoU (Spatio-Temporal IoU) giữa predicted volume và ground truth volume

    Args:
        pred_bboxes: List of predicted bboxes với 'frame' key
        gt_bboxes: List of ground truth bboxes với 'frame' key

    Returns:
        ST-IoU score (0 to 1)
    """
    if not pred_bboxes or not gt_bboxes:
        return 0.0

    # Tạo tập hợp frames
    pred_frames = {bbox['frame'] for bbox in pred_bboxes}
    gt_frames = {bbox['frame'] for bbox in gt_bboxes}

    # Frames chung (intersection)
    common_frames = pred_frames & gt_frames

    if not common_frames:
        return 0.0

    # Tính spatial overlap cho các frames chung
    spatial_overlap_volume = 0.0
    total_volume = 0.0

    # Tính từ predicted bboxes
    for pred_bbox in pred_bboxes:
        frame = pred_bbox['frame']
        if frame in gt_frames:
            # Tìm bbox tương ứng ở ground truth
            matching_gt_bboxes = [
                bbox for bbox in gt_bboxes
                if bbox['frame'] == frame
            ]

            if matching_gt_bboxes:
                # Tính max spatial IoU với bất kỳ ground truth nào
                pred_bbox_coords = (pred_bbox['x1'], pred_bbox['y1'], pred_bbox['x2'], pred_bbox['y2'])
                max_spatial_iou = 0.0

                for gt_bbox in matching_gt_bboxes:
                    gt_bbox_coords = (gt_bbox['x1'], gt_bbox['y1'], gt_bbox['x2'], gt_bbox['y2'])
                    spatial_iou = calculate_spatial_iou(pred_bbox_coords, gt_bbox_coords)
                    max_spatial_iou = max(max_spatial_iou, spatial_iou)

                spatial_overlap_volume += max_spatial_iou
            total_volume += 1

    # Tính từ ground truth bboxes (để không miss)
    for gt_bbox in gt_bboxes:
        frame = gt_bbox['frame']
        if frame in pred_frames:
            matching_pred_bboxes = [
                bbox for bbox in pred_bboxes
                if bbox['frame'] == frame
            ]

            if not matching_pred_bboxes:
                total_volume += 1

    # ST-IoU = (spatial intersection volume) / (spatial union volume)
    temporal_iou = calculate_temporal_iou(pred_frames, gt_frames)

    # ST-IoU kết hợp spatial và temporal
    if total_volume > 0:
        spatial_accuracy = spatial_overlap_volume / total_volume
    else:
        spatial_accuracy = 0.0

    st_iou = spatial_accuracy * temporal_iou

    return st_iou


def load_ground_truth(annotation_path: str) -> Dict:
    """
    Load ground truth annotations từ JSON file

    Args:
        annotation_path: Đường dẫn tới annotations.json

    Returns:
        Dictionary chứa ground truth data
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    gt_dict = {}
    for item in annotations:
        video_id = item['video_id']
        # Mỗi video có thể có multiple objects (annotations)
        gt_dict[video_id] = item.get('annotations', [])

    return gt_dict


def load_predictions(results_dir: str) -> Dict:
    """
    Load predictions từ results directory

    Args:
        results_dir: Đường dẫn tới thư mục chứa kết quả

    Returns:
        Dictionary chứa predictions {video_id: list of all bboxes}
    """
    predictions = {}

    if not os.path.exists(results_dir):
        print(f"⚠️ Results directory not found: {results_dir}")
        return predictions

    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json') and not filename.startswith('evaluation'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    video_id = data.get('video_id')
                    if video_id:
                        # Parse detections và flatten bboxes
                        all_bboxes = []
                        detections = data.get('detections', [])

                        # Nếu detections là danh sách các objects, mỗi object có bboxes
                        if isinstance(detections, list):
                            for detection in detections:
                                if isinstance(detection, dict):
                                    # Format từ pipeline: {"bboxes": [...]}
                                    if 'bboxes' in detection:
                                        for bbox in detection.get('bboxes', []):
                                            all_bboxes.append(bbox)
                                    # Hoặc format khác: detection là bbox trực tiếp
                                    elif 'frame' in detection:
                                        all_bboxes.append(detection)

                        predictions[video_id] = all_bboxes
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")

    return predictions


def evaluate_video(
    video_id: str,
    predictions: List[Dict],
    ground_truth_annotations: List[Dict]
) -> Dict:
    """
    Đánh giá một video sử dụng ST-IoU

    Args:
        video_id: ID của video
        predictions: Danh sách dự đoán
        ground_truth_annotations: Danh sách ground truth (có thể nhiều objects)

    Returns:
        Dictionary chứa metrics cho video này
    """
    # Nếu không có predictions hoặc ground truth
    if not ground_truth_annotations:
        st_iou_scores = []
    elif not predictions:
        st_iou_scores = [0.0] * len(ground_truth_annotations)
    else:
        # Tính ST-IoU với mỗi ground truth object
        st_iou_scores = []
        for gt_obj in ground_truth_annotations:
            gt_bboxes = gt_obj.get('bboxes', [])
            if gt_bboxes:
                st_iou = calculate_st_iou(predictions, gt_bboxes)
                st_iou_scores.append(st_iou)

    # Lấy mean ST-IoU cho video này
    mean_st_iou = np.mean(st_iou_scores) if st_iou_scores else 0.0

    # Tính thêm các metrics khác
    num_gt_objects = len(ground_truth_annotations)
    num_predictions = len(predictions)

    # Tính số frame được detect
    pred_frames = {p.get('frame') for p in predictions}
    all_gt_frames = set()
    for gt_obj in ground_truth_annotations:
        for bbox in gt_obj.get('bboxes', []):
            all_gt_frames.add(bbox['frame'])

    common_frames = len(pred_frames & all_gt_frames)
    total_gt_frames = len(all_gt_frames)
    frame_recall = common_frames / total_gt_frames if total_gt_frames > 0 else 0.0

    metrics = {
        'video_id': video_id,
        'st_iou': mean_st_iou,
        'per_object_st_iou': st_iou_scores,
        'num_gt_objects': num_gt_objects,
        'num_predictions': num_predictions,
        'num_gt_frames': total_gt_frames,
        'num_detected_frames': common_frames,
        'frame_recall': frame_recall,
    }

    return metrics


def evaluate_all(
    annotations_path: str,
    results_dir: str
) -> Dict:
    """
    Đánh giá toàn bộ dataset sử dụng ST-IoU

    Args:
        annotations_path: Đường dẫn tới annotations.json
        results_dir: Đường dẫn tới thư mục kết quả

    Returns:
        Dictionary chứa tổng hợp metrics
    """
    print(f"\n📊 Spatio-Temporal IoU (ST-IoU) Evaluation")
    print("=" * 80)

    # Load ground truth và predictions
    print(f"\n📂 Loading ground truth from {annotations_path}...")
    ground_truth = load_ground_truth(annotations_path)
    print(f"✅ Loaded {len(ground_truth)} videos")

    print(f"\n📂 Loading predictions from {results_dir}...")
    predictions = load_predictions(results_dir)
    print(f"✅ Loaded {len(predictions)} videos with predictions")

    # Đánh giá từng video
    print(f"\n🔍 Evaluating each video with ST-IoU...")
    all_metrics = []

    for video_id in tqdm(sorted(ground_truth.keys()), desc="Evaluating"):
        gt = ground_truth.get(video_id, [])
        preds = predictions.get(video_id, [])

        metrics = evaluate_video(
            video_id,
            preds,
            gt
        )
        all_metrics.append(metrics)

    # Tính tổng hợp
    print("\n" + "=" * 80)
    print("📈 Overall Results (ST-IoU Metric)")
    print("=" * 80)

    st_iou_scores = [m['st_iou'] for m in all_metrics]
    mean_st_iou = np.mean(st_iou_scores)
    std_st_iou = np.std(st_iou_scores)
    min_st_iou = np.min(st_iou_scores)
    max_st_iou = np.max(st_iou_scores)

    total_gt_objects = sum(m['num_gt_objects'] for m in all_metrics)
    total_predictions = sum(m['num_predictions'] for m in all_metrics)
    total_gt_frames = sum(m['num_gt_frames'] for m in all_metrics)
    total_detected_frames = sum(m['num_detected_frames'] for m in all_metrics)

    print(f"\n📊 ST-IoU Statistics:")
    print(f"  Mean ST-IoU: {mean_st_iou:.4f}")
    print(f"  Std Dev: {std_st_iou:.4f}")
    print(f"  Min ST-IoU: {min_st_iou:.4f}")
    print(f"  Max ST-IoU: {max_st_iou:.4f}")

    print(f"\n📈 Detection Statistics:")
    print(f"  Total Ground Truth Objects: {total_gt_objects}")
    print(f"  Total Predictions: {total_predictions}")
    print(f"  Ground Truth Frames: {total_gt_frames}")
    print(f"  Detected Frames: {total_detected_frames}")

    frame_recall = total_detected_frames / total_gt_frames if total_gt_frames > 0 else 0.0
    print(f"  Frame Recall: {frame_recall:.4f} ({frame_recall*100:.2f}%)")

    # In chi tiết từng video
    print(f"\n📋 Per-Video ST-IoU Results:")
    print("-" * 80)
    print(f"{'Video ID':<20} {'ST-IoU':<10} {'Objects':<10} {'Preds':<10} {'Frame Recall':<15}")
    print("-" * 80)

    for m in sorted(all_metrics, key=lambda x: x['st_iou'], reverse=True):
        print(f"{m['video_id']:<20} {m['st_iou']:<10.4f} {m['num_gt_objects']:<10} "
              f"{m['num_predictions']:<10} {m['frame_recall']:<15.4f}")

    print("-" * 80)
    print(f"{'AVERAGE':<20} {mean_st_iou:<10.4f}")

    # Lưu kết quả chi tiết
    results_summary = {
        'metric_type': 'ST-IoU (Spatio-Temporal IoU)',
        'overall_metrics': {
            'mean_st_iou': float(mean_st_iou),
            'std_st_iou': float(std_st_iou),
            'min_st_iou': float(min_st_iou),
            'max_st_iou': float(max_st_iou),
            'frame_recall': float(frame_recall),
            'total_gt_objects': int(total_gt_objects),
            'total_predictions': int(total_predictions),
            'total_gt_frames': int(total_gt_frames),
            'total_detected_frames': int(total_detected_frames),
        },
        'per_video_metrics': all_metrics
    }

    return results_summary


def save_evaluation_results(summary: Dict, output_path: str) -> None:
    """
    Lưu kết quả đánh giá vào file JSON

    Args:
        summary: Dictionary chứa tổng hợp metrics
        output_path: Đường dẫn file output
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    import sys

    # Các đường dẫn mặc định
    dataset_root = '/Users/huuthieu/Desktop/learning/zaloAI/train'
    annotations_path = os.path.join(dataset_root, 'annotations', 'annotations.json')
    results_dir = './results'

    # Cho phép override từ command line
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    if len(sys.argv) > 2:
        annotations_path = sys.argv[2]

    print(f"🚁 Drone Object Detection - ST-IoU Evaluation")
    print("=" * 80)
    print(f"Annotations: {annotations_path}")
    print(f"Results: {results_dir}")

    # Chạy đánh giá
    summary = evaluate_all(
        annotations_path=annotations_path,
        results_dir=results_dir
    )

    # Lưu kết quả
    output_path = os.path.join(results_dir, 'evaluation_results.json')
    save_evaluation_results(summary, output_path)

    print("\n✅ Evaluation complete!")
