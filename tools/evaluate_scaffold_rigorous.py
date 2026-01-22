#!/usr/bin/env python3
"""
Rigorous Academic Evaluation for Scaffold Missing Detection - FIXED VERSION

Properly aligned with data generation code structure.

Data Structure (from question_generator.py):
- test_questions.jsonl: {question_id, point, text: <question>, category}
- test_gt.jsonl: {question_id, point, text: <answer>, label, bboxes, task_type}
- predictions: {question_id, text: <model_output>}

Usage:
    python tools/evaluate_scaffold_rigorous.py \
        --predictions ./outputs/test_answers.jsonl \
        --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
        --output-dir ./evaluation_results
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

import numpy as np


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BBox3D:
    """3D bounding box represented by 8 corners."""
    corners: np.ndarray

    @classmethod
    def from_list(cls, corners_list: List[List[float]]) -> Optional['BBox3D']:
        try:
            arr = np.array(corners_list)
            if arr.shape == (8, 3):
                return cls(corners=arr)
        except:
            pass
        return None

    def get_axis_aligned_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.corners.min(axis=0), self.corners.max(axis=0)

    def volume(self) -> float:
        min_b, max_b = self.get_axis_aligned_bounds()
        dims = max_b - min_b
        return float(np.prod(np.maximum(dims, 0)))

    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


@dataclass
class DetailedMetrics:
    """Comprehensive evaluation metrics."""
    # Binary Classification
    binary_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Confusion Matrix
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    # Component Type Detection (from answer text)
    component_detection_accuracy: float = 0.0
    vertical_detection_rate: float = 0.0
    horizontal_detection_rate: float = 0.0
    platform_detection_rate: float = 0.0

    # Count Accuracy
    count_exact_accuracy: float = 0.0
    count_tolerance1_accuracy: float = 0.0
    count_mae: float = 0.0

    # BBox Grounding
    bbox_iou_mean: float = 0.0
    bbox_iou_at_50: float = 0.0
    bbox_iou_at_25: float = 0.0
    bbox_detection_rate: float = 0.0
    bbox_false_positive_rate: float = 0.0

    # Per-Task Accuracy
    per_task_accuracy: Dict[str, float] = field(default_factory=dict)

    # Sample counts
    num_samples: int = 0
    num_yes_gt: int = 0
    num_no_gt: int = 0
    num_pred_boxes: int = 0
    num_gt_boxes: int = 0


@dataclass
class AblationResult:
    """Results from ablation study."""
    condition: str
    binary_accuracy: float
    expected_range: Tuple[float, float]
    passed: bool
    interpretation: str


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    main_metrics: DetailedMetrics
    ablation_results: List[AblationResult]
    data_leakage_detected: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# Parsing Functions - Aligned with question_generator.py output
# ============================================================================

def parse_binary_from_prediction(pred_text: str, task_type: str) -> str:
    """
    Parse Yes/No from model prediction text.

    The question types and their Yes/No meanings:
    - missing_detection_summary: Yes = missing exists, No = no missing
    - missing_detection_floor: Yes = missing on floor, No = no missing
    - missing_detection_bay: Yes = missing in bay, No = no missing
    - missing_detection_specific: Yes = component present, No = component missing
    - missing_detection_vertical_summary: Yes = vertical missing, No = no vertical missing
    - missing_detection_horizontal_summary: Yes = horizontal missing, No = no horizontal missing
    """
    text_lower = pred_text.lower().strip()

    # For specific component questions, the meaning is reversed
    # "Is there a component at X?" -> Yes = present, No = missing
    if task_type == 'missing_detection_specific':
        # Check if model says component exists
        if text_lower.startswith('yes'):
            return 'Yes'  # Component is present
        elif text_lower.startswith('no'):
            return 'No'   # Component is missing
        # Inference
        if 'missing' in text_lower or 'absent' in text_lower:
            return 'No'
        if 'present' in text_lower or 'exists' in text_lower or 'there is' in text_lower:
            return 'Yes'
        return 'No'  # Default: component not found

    # For all other question types: Yes = missing detected, No = no missing
    # Explicit start patterns
    if text_lower.startswith('yes'):
        # Double check for negation
        if 'no missing' in text_lower[:50] or 'no defect' in text_lower[:50]:
            return 'No'
        return 'Yes'
    elif text_lower.startswith('no'):
        # "No missing" = No (correct)
        # But "No, there are missing" would be Yes
        if 'missing' in text_lower[3:30] and 'no missing' not in text_lower:
            return 'Yes'
        return 'No'

    # Inference from content
    # Strong negative indicators (check first)
    negative_phrases = [
        'no missing', 'no defect', 'properly installed',
        'all present', 'all components are present', 'complete structure',
        'no component', 'all elements'
    ]
    for phrase in negative_phrases:
        if phrase in text_lower:
            return 'No'

    # Positive indicators
    positive_phrases = [
        'missing', 'detected', 'found', 'absent', 'lack',
        'defect', 'not present', 'is missing'
    ]
    for phrase in positive_phrases:
        if phrase in text_lower:
            return 'Yes'

    return 'No'  # Default to no missing


def parse_missing_count_from_text(text: str) -> int:
    """
    Parse the number of missing components from answer text.

    Expected format from question_generator.py:
    "Missing components detected (N total):"
    "Missing: N vertical post(s):"
    "N component(s) are missing"
    """
    text_lower = text.lower()

    # Pattern 1: "Missing components detected (N total)"
    match = re.search(r'missing[^(]*\((\d+)\s*total\)', text_lower)
    if match:
        return int(match.group(1))

    # Pattern 2: "N component(s) are missing"
    match = re.search(r'(\d+)\s*component[s]?\s*(?:are|is)?\s*missing', text_lower)
    if match:
        return int(match.group(1))

    # Pattern 3: Count bullet points "- Vertical post at..."
    bullet_count = len(re.findall(r'^-\s+(?:vertical|horizontal|platform)', text_lower, re.MULTILINE))
    if bullet_count > 0:
        return bullet_count

    # Pattern 4: "Missing: N"
    match = re.search(r'missing:\s*(\d+)', text_lower)
    if match:
        return int(match.group(1))

    return 0


def parse_component_types_from_text(text: str) -> Dict[str, int]:
    """
    Parse component types mentioned as missing from text.

    Expected format:
    "- Vertical post at column X, row Y: [[bbox]]"
    "- Horizontal beam (X) at floor Z, bay W: [[bbox]]"
    "- Platform at floor Z, bay W: [[bbox]]"
    """
    counts = {'vertical': 0, 'horizontal': 0, 'platform': 0}
    text_lower = text.lower()

    # Count bullet point mentions
    counts['vertical'] = len(re.findall(r'-\s*vertical\s+post', text_lower))
    counts['horizontal'] = len(re.findall(r'-\s*horizontal\s+beam', text_lower))
    counts['platform'] = len(re.findall(r'-\s*platform\s+at', text_lower))

    return counts


def parse_bboxes_from_text(text: str) -> List[List[List[float]]]:
    """
    Extract 3D bounding boxes from text.

    Expected format: [[x,y,z], [x,y,z], [x,y,z], [x,y,z], [x,y,z], [x,y,z], [x,y,z], [x,y,z]]
    """
    bboxes = []

    # Pattern for 8-corner bbox
    # More flexible pattern to handle whitespace variations
    bbox_pattern = r'\[\s*\[[\s\d.,-]+\](?:\s*,\s*\[[\s\d.,-]+\]){7}\s*\]'

    matches = re.findall(bbox_pattern, text)

    for match in matches:
        try:
            # Clean up and parse
            clean = re.sub(r'\s+', '', match)
            bbox = json.loads(clean)
            if len(bbox) == 8 and all(len(corner) == 3 for corner in bbox):
                bboxes.append(bbox)
        except (json.JSONDecodeError, ValueError):
            continue

    return bboxes


# ============================================================================
# IoU Computation
# ============================================================================

def compute_iou_3d(box1: BBox3D, box2: BBox3D) -> float:
    """Compute 3D IoU using axis-aligned bounding box approximation."""
    min1, max1 = box1.get_axis_aligned_bounds()
    min2, max2 = box2.get_axis_aligned_bounds()

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    if np.any(inter_max <= inter_min):
        return 0.0

    inter_vol = float(np.prod(inter_max - inter_min))
    union_vol = box1.volume() + box2.volume() - inter_vol

    return inter_vol / union_vol if union_vol > 0 else 0.0


def match_bboxes(
    pred_bboxes: List[List[List[float]]],
    gt_bboxes: List[List[List[float]]],
    iou_threshold: float = 0.25
) -> Tuple[List[float], int, int]:
    """
    Match predicted boxes to ground truth using greedy algorithm.

    Returns: (all_ious, num_true_positive, num_false_positive)
    """
    if not gt_bboxes:
        # No GT boxes, all predictions are false positives
        return [], 0, len(pred_bboxes)

    if not pred_bboxes:
        # No predictions, all GT boxes are missed
        return [], 0, 0

    pred_boxes = [BBox3D.from_list(b) for b in pred_bboxes]
    gt_boxes = [BBox3D.from_list(b) for b in gt_bboxes]

    # Filter out invalid boxes
    pred_boxes = [b for b in pred_boxes if b is not None]
    gt_boxes = [b for b in gt_boxes if b is not None]

    if not pred_boxes or not gt_boxes:
        return [], 0, len(pred_bboxes)

    matched_gt = set()
    all_ious = []
    num_tp = 0

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = compute_iou_3d(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        all_ious.append(best_iou)
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            num_tp += 1

    num_fp = len(pred_boxes) - num_tp
    return all_ious, num_tp, num_fp


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_detailed(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> DetailedMetrics:
    """
    Compute comprehensive evaluation metrics.

    Uses GT's label field directly (not parsed from text).
    Parses predictions from model output text.
    """
    gt_lookup = {gt['question_id']: gt for gt in ground_truth}

    metrics = DetailedMetrics()

    # Accumulators
    tp, tn, fp, fn = 0, 0, 0, 0

    # Component detection tracking
    comp_gt_counts = {'vertical': 0, 'horizontal': 0, 'platform': 0}
    comp_pred_correct = {'vertical': 0, 'horizontal': 0, 'platform': 0}

    # Count tracking
    count_exact = 0
    count_tol1 = 0
    count_errors = []
    count_samples = 0

    # BBox tracking
    all_ious = []
    total_tp_boxes = 0
    total_fp_boxes = 0
    total_gt_boxes = 0
    total_pred_boxes = 0

    # Per-task tracking
    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    for pred in predictions:
        qid = pred.get('question_id')
        if qid not in gt_lookup:
            continue

        gt = gt_lookup[qid]

        # GT values (directly from data, no parsing needed)
        gt_label = gt.get('label', 'No')
        gt_bboxes = gt.get('bboxes', [])
        gt_answer_text = gt.get('text', '')
        task_type = gt.get('task_type', 'unknown')

        # Prediction values (parsed from model output)
        pred_text = pred.get('text', '')
        pred_label = parse_binary_from_prediction(pred_text, task_type)
        pred_bboxes = parse_bboxes_from_text(pred_text)

        metrics.num_samples += 1
        task_total[task_type] += 1

        # ============ Binary Classification ============
        if pred_label == gt_label:
            task_correct[task_type] += 1

        # Confusion matrix (Yes = positive = missing detected)
        if gt_label == 'Yes':
            metrics.num_yes_gt += 1
            if pred_label == 'Yes':
                tp += 1
            else:
                fn += 1
        else:
            metrics.num_no_gt += 1
            if pred_label == 'No':
                tn += 1
            else:
                fp += 1

        # ============ Component Type Detection ============
        # Only for summary-type questions with Yes label
        if gt_label == 'Yes' and 'summary' in task_type:
            gt_comp = parse_component_types_from_text(gt_answer_text)
            pred_comp = parse_component_types_from_text(pred_text)

            for comp_type in ['vertical', 'horizontal', 'platform']:
                gt_count = gt_comp[comp_type]
                pred_count = pred_comp[comp_type]

                if gt_count > 0:
                    comp_gt_counts[comp_type] += gt_count
                    # Count how many the model correctly identified
                    comp_pred_correct[comp_type] += min(pred_count, gt_count)

        # ============ Count Accuracy ============
        if gt_label == 'Yes':
            gt_count = parse_missing_count_from_text(gt_answer_text)
            pred_count = parse_missing_count_from_text(pred_text)

            if gt_count > 0:
                count_samples += 1
                error = abs(pred_count - gt_count)
                count_errors.append(error)

                if error == 0:
                    count_exact += 1
                if error <= 1:
                    count_tol1 += 1

        # ============ BBox IoU ============
        total_pred_boxes += len(pred_bboxes)
        total_gt_boxes += len(gt_bboxes)

        if gt_bboxes:
            ious, tp_boxes, fp_boxes = match_bboxes(pred_bboxes, gt_bboxes)
            all_ious.extend(ious)
            total_tp_boxes += tp_boxes
            total_fp_boxes += fp_boxes

    # ============ Calculate Final Metrics ============
    n = metrics.num_samples

    # Binary classification
    metrics.true_positive = tp
    metrics.true_negative = tn
    metrics.false_positive = fp
    metrics.false_negative = fn

    metrics.binary_accuracy = ((tp + tn) / n * 100) if n > 0 else 0
    metrics.precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    metrics.recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0

    if metrics.precision + metrics.recall > 0:
        metrics.f1_score = (2 * metrics.precision * metrics.recall /
                           (metrics.precision + metrics.recall))

    # Component detection rates
    for comp_type in ['vertical', 'horizontal', 'platform']:
        if comp_gt_counts[comp_type] > 0:
            rate = comp_pred_correct[comp_type] / comp_gt_counts[comp_type] * 100
            setattr(metrics, f'{comp_type}_detection_rate', rate)

    total_comp_gt = sum(comp_gt_counts.values())
    total_comp_correct = sum(comp_pred_correct.values())
    if total_comp_gt > 0:
        metrics.component_detection_accuracy = total_comp_correct / total_comp_gt * 100

    # Count accuracy
    if count_samples > 0:
        metrics.count_exact_accuracy = count_exact / count_samples * 100
        metrics.count_tolerance1_accuracy = count_tol1 / count_samples * 100
    if count_errors:
        metrics.count_mae = np.mean(count_errors)

    # BBox metrics
    metrics.num_pred_boxes = total_pred_boxes
    metrics.num_gt_boxes = total_gt_boxes

    if all_ious:
        metrics.bbox_iou_mean = np.mean(all_ious) * 100
        metrics.bbox_iou_at_50 = sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious) * 100
        metrics.bbox_iou_at_25 = sum(1 for iou in all_ious if iou >= 0.25) / len(all_ious) * 100

    if total_gt_boxes > 0:
        metrics.bbox_detection_rate = total_tp_boxes / total_gt_boxes * 100
    if total_pred_boxes > 0:
        metrics.bbox_false_positive_rate = total_fp_boxes / total_pred_boxes * 100

    # Per-task accuracy
    metrics.per_task_accuracy = {
        task: (task_correct[task] / task_total[task] * 100) if task_total[task] > 0 else 0
        for task in task_total
    }

    return metrics


# ============================================================================
# Ablation Study Support
# ============================================================================

def check_data_leakage(
    main_metrics: DetailedMetrics,
    ablation_noise_acc: Optional[float] = None
) -> Tuple[bool, List[str]]:
    """
    Check for data leakage indicators.

    Red flags:
    1. Very high binary accuracy (>95%) with very low IoU (<5%)
    2. Noise ablation accuracy significantly above 50%
    """
    warnings = []
    leakage_detected = False

    # Check 1: High accuracy with low IoU mismatch
    if main_metrics.binary_accuracy > 95 and main_metrics.bbox_iou_at_25 < 5:
        leakage_detected = True
        warnings.append(
            f"DATA LEAKAGE SUSPECTED: Binary accuracy is {main_metrics.binary_accuracy:.1f}% "
            f"but BBox IoU@0.25 is only {main_metrics.bbox_iou_at_25:.1f}%. "
            f"Model may be using text shortcuts instead of analyzing point clouds."
        )

    # Check 2: Noise ablation
    if ablation_noise_acc is not None and ablation_noise_acc > 60:
        leakage_detected = True
        warnings.append(
            f"DATA LEAKAGE CONFIRMED: Noise ablation accuracy is {ablation_noise_acc:.1f}%, "
            f"expected ~50% for random point clouds. Model is not using visual information."
        )

    # Check 3: Suspiciously perfect accuracy
    if main_metrics.binary_accuracy > 99:
        warnings.append(
            f"WARNING: Binary accuracy of {main_metrics.binary_accuracy:.1f}% is suspiciously high. "
            f"Please verify with ablation studies."
        )

    return leakage_detected, warnings


# ============================================================================
# Reporting
# ============================================================================

def print_evaluation_report(report: EvaluationReport):
    """Print comprehensive evaluation report."""
    m = report.main_metrics

    print("\n" + "=" * 80)
    print(" SCAFFOLD MISSING DETECTION - RIGOROUS EVALUATION REPORT")
    print(f" Generated: {report.timestamp}")
    print("=" * 80)

    # Data Leakage Status
    if report.data_leakage_detected:
        print("\n⚠️  DATA LEAKAGE DETECTED - Results may be invalid!")
    else:
        print("\n✓ No obvious data leakage detected")

    # Binary Classification
    print("\n" + "-" * 50)
    print(" 1. BINARY CLASSIFICATION")
    print("-" * 50)
    print(f"  Accuracy:  {m.binary_accuracy:.2f}%")
    print(f"  Precision: {m.precision:.2f}%")
    print(f"  Recall:    {m.recall:.2f}%")
    print(f"  F1 Score:  {m.f1_score:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Yes    No")
    print(f"    Actual Yes {m.true_positive:4d}  {m.false_negative:4d}")
    print(f"           No  {m.false_positive:4d}  {m.true_negative:4d}")
    print(f"\n  GT Distribution: Yes={m.num_yes_gt}, No={m.num_no_gt}")

    # Component Detection
    print("\n" + "-" * 50)
    print(" 2. COMPONENT TYPE DETECTION")
    print("-" * 50)
    print(f"  Overall:    {m.component_detection_accuracy:.2f}%")
    print(f"  Vertical:   {m.vertical_detection_rate:.2f}%")
    print(f"  Horizontal: {m.horizontal_detection_rate:.2f}%")
    print(f"  Platform:   {m.platform_detection_rate:.2f}%")

    # Count Accuracy
    print("\n" + "-" * 50)
    print(" 3. COUNT ACCURACY")
    print("-" * 50)
    print(f"  Exact Match: {m.count_exact_accuracy:.2f}%")
    print(f"  ±1 Tolerance: {m.count_tolerance1_accuracy:.2f}%")
    print(f"  Mean Abs Error: {m.count_mae:.2f}")

    # BBox Grounding
    print("\n" + "-" * 50)
    print(" 4. BOUNDING BOX GROUNDING")
    print("-" * 50)
    print(f"  Mean IoU:       {m.bbox_iou_mean:.2f}%")
    print(f"  IoU@0.50:       {m.bbox_iou_at_50:.2f}%")
    print(f"  IoU@0.25:       {m.bbox_iou_at_25:.2f}%")
    print(f"  Detection Rate: {m.bbox_detection_rate:.2f}%")
    print(f"  False Pos Rate: {m.bbox_false_positive_rate:.2f}%")
    print(f"  (Predicted: {m.num_pred_boxes}, GT: {m.num_gt_boxes})")

    # Per-Task
    if m.per_task_accuracy:
        print("\n" + "-" * 50)
        print(" 5. PER-TASK BINARY ACCURACY")
        print("-" * 50)
        for task, acc in sorted(m.per_task_accuracy.items()):
            print(f"  {task}: {acc:.2f}%")

    # Warnings
    if report.warnings:
        print("\n" + "-" * 50)
        print(" ⚠️  WARNINGS")
        print("-" * 50)
        for w in report.warnings:
            print(f"  • {w}")

    print("\n" + "=" * 80)


def save_results(report: EvaluationReport, output_dir: str):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON report
    report_dict = {
        'timestamp': report.timestamp,
        'data_leakage_detected': report.data_leakage_detected,
        'main_metrics': asdict(report.main_metrics),
        'ablation_results': [asdict(a) for a in report.ablation_results],
        'warnings': report.warnings,
        'recommendations': report.recommendations
    }

    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)

    # LaTeX table
    m = report.main_metrics
    latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Scaffold Missing Detection Results}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Binary Accuracy & {m.binary_accuracy:.2f}\\% \\\\
Precision & {m.precision:.2f}\\% \\\\
Recall & {m.recall:.2f}\\% \\\\
F1 Score & {m.f1_score:.2f}\\% \\\\
\\midrule
Component Detection & {m.component_detection_accuracy:.2f}\\% \\\\
Count Exact Match & {m.count_exact_accuracy:.2f}\\% \\\\
\\midrule
BBox IoU@0.50 & {m.bbox_iou_at_50:.2f}\\% \\\\
BBox IoU@0.25 & {m.bbox_iou_at_25:.2f}\\% \\\\
BBox Detection Rate & {m.bbox_detection_rate:.2f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write(latex)

    print(f"\nResults saved to: {output_dir}")


# ============================================================================
# Utility
# ============================================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rigorous Academic Evaluation')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSONL')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth JSONL')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory')
    parser.add_argument('--ablation-noise-predictions', type=str, default=None,
                       help='Path to noise ablation predictions for leakage check')

    args = parser.parse_args()

    print("=" * 80)
    print(" RIGOROUS ACADEMIC EVALUATION - FIXED VERSION")
    print("=" * 80)

    # Load data
    print(f"\nLoading predictions: {args.predictions}")
    predictions = load_jsonl(args.predictions)

    print(f"Loading ground truth: {args.ground_truth}")
    ground_truth = load_jsonl(args.ground_truth)

    print(f"Loaded {len(predictions)} predictions, {len(ground_truth)} ground truth")

    # Main evaluation
    print("\nRunning evaluation...")
    main_metrics = evaluate_detailed(predictions, ground_truth)

    # Ablation check
    ablation_results = []
    ablation_noise_acc = None

    if args.ablation_noise_predictions and os.path.exists(args.ablation_noise_predictions):
        print("\nRunning noise ablation check...")
        noise_preds = load_jsonl(args.ablation_noise_predictions)
        noise_metrics = evaluate_detailed(noise_preds, ground_truth)
        ablation_noise_acc = noise_metrics.binary_accuracy

        ablation_results.append(AblationResult(
            condition='noise',
            binary_accuracy=ablation_noise_acc,
            expected_range=(45, 55),
            passed=45 <= ablation_noise_acc <= 55,
            interpretation='Random noise should yield ~50% accuracy'
        ))

    # Check for data leakage
    leakage_detected, warnings = check_data_leakage(main_metrics, ablation_noise_acc)

    # Create report
    report = EvaluationReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        main_metrics=main_metrics,
        ablation_results=ablation_results,
        data_leakage_detected=leakage_detected,
        warnings=warnings
    )

    # Output
    print_evaluation_report(report)
    save_results(report, args.output_dir)

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
