"""
Evaluation script for scaffold missing platform detection.

This script compares model predictions against ground truth labels for
the missing‑platform detection task.  It expects two JSONL files:

    * ``answers_file``: produced by running ``llava.eval.model_vqa``
      (or similar) on ``question_missing.jsonl``.  Each line should
      contain a JSON object with at least the fields ``question_id`` and
      ``text`` (the predicted answer).

    * ``gt_file``: ground truth file created by
      ``missing_detection_dataset.py``.  Each line contains a JSON object
      with ``question_id`` and ``text`` fields; the ``text`` field is
      either ``"Yes"`` or ``"No"`` indicating whether the scene has any
      missing platforms.

The script prints overall accuracy and a simple confusion matrix.
"""

import argparse
import json
from collections import Counter


def evaluate_missing_detection(answers_file: str, gt_file: str) -> None:
    # Load ground truth labels
    gt_labels = {}
    with open(gt_file, 'r', encoding='utf-8') as f_gt:
        for line in f_gt:
            record = json.loads(line)
            qid = record['question_id']
            # Use 'label' field if available (Idea 1 format), otherwise fallback to 'text'
            if 'label' in record:
                label = record['label'].strip().lower()
            else:
                label = record['text'].strip().lower()
            gt_labels[qid] = label

    # Evaluate predictions
    total = 0
    correct = 0
    confusion = Counter()

    with open(answers_file, 'r', encoding='utf-8') as f_ans:
        for line in f_ans:
            ans = json.loads(line)
            qid = ans.get('question_id')
            pred_text = ans.get('text', '').strip().lower()
            # Map free‑text answers to yes/no labels.  A simple heuristic:
            if 'yes' in pred_text:
                pred_label = 'yes'
            elif 'no' in pred_text:
                pred_label = 'no'
            else:
                pred_label = 'unknown'
            # Compare with ground truth
            gt_label = gt_labels.get(qid)
            if gt_label is None:
                continue
            total += 1
            confusion[(gt_label, pred_label)] += 1
            if pred_label == gt_label:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Evaluated {total} samples")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion matrix (gt,pred -> count):")
    for key, count in confusion.items():
        print(f"  {key}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate missing platform detection predictions.')
    parser.add_argument('--answers-file', type=str, required=True,
                        help='Path to model prediction JSONL file.')
    parser.add_argument('--gt-file', type=str, required=True,
                        help='Path to ground truth JSONL file.')
    args = parser.parse_args()
    evaluate_missing_detection(args.answers_file, args.gt_file)