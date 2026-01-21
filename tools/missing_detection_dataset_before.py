"""
Dataset generator for scaffold missing‑component detection (final version).

This script synthesizes scaffold scenes using the ``EnhancedScaffoldGeneratorFinal`` and
creates both training and validation splits for missing detection tasks.  It
supports configurable train/validation ratios and includes bounding box
annotations in the evaluation ground truth.

Outputs:

* A **training SFT JSON** (``scaffold_missing_sft.json``) containing conversational
  examples for supervised fine‑tuning.  Only missing detection questions are
  included, and only scenes in the training split contribute to this file.
* Two **validation files** for evaluating model performance:
  - ``question_missing.jsonl``: a list of questions without the ``<point>`` token.
  - ``gt_missing.jsonl``: ground truth labels (``Yes``/``No``) and 3D bounding boxes
    for the relevant components.  Bounding boxes are populated for missing
    detection tasks where applicable.

Usage example:

    python missing_detection_dataset_final.py \
        --num-scenes 500 \
        --train-ratio 0.8 \
        --val-ratio 0.2 \
        --output-dir ./playground/data/shapellm/scaffold_missing

"""

import argparse
import json
import os
import random
from typing import List, Dict, Any

import numpy as np  # type: ignore

from modified_scaffold_generator_final import EnhancedScaffoldGeneratorFinal


def generate_missing_dataset(
    output_dir: str,
    num_scenes: int = 100,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> None:
    """Generate synthetic scenes and export missing detection datasets.

    Scenes are split into training and validation subsets according to the
    provided ratios.  Only training scenes contribute to the SFT JSON, and
    only validation scenes produce questions and ground truth entries.

    Args:
        output_dir: Destination directory for dataset files.
        num_scenes: Number of scenes to synthesize.
        train_ratio: Fraction of scenes used for training.  Must be between 0 and 1.
        val_ratio: Fraction of scenes used for validation.  train_ratio + val_ratio <= 1.
    """

    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio and val_ratio must be non‑negative and sum to <= 1.0")

    os.makedirs(output_dir, exist_ok=True)
    pcs_dir = os.path.join(output_dir, 'pcs')
    os.makedirs(pcs_dir, exist_ok=True)

    scene_records: List[Dict[str, Any]] = []
    generator = EnhancedScaffoldGeneratorFinal(random_seed=None)

    # Generate all scenes first
    for i in range(num_scenes):
        scene_id = f"scaffold_{i:05d}"
        scene_data = generator.generate_scene_data(scene_id)
        if scene_data is None:
            continue
        # Save point cloud
        coord = scene_data['coord']
        npy_path = os.path.join(pcs_dir, f"{scene_id}.npy")
        np.save(npy_path, coord.astype('float32'))
        # Store annotations and components for later splitting
        scene_records.append({
            'scene_id': scene_id,
            'annotations': scene_data['annotations'],
            'components': scene_data['components'],
        })

    # Shuffle scenes and split
    random.shuffle(scene_records)
    n_train = int(train_ratio * len(scene_records))
    n_val = int(val_ratio * len(scene_records))
    train_records = scene_records[:n_train]
    val_records = scene_records[n_train:n_train + n_val]

    sft_entries: List[Dict[str, Any]] = []
    eval_questions: List[Dict[str, Any]] = []
    eval_gt: List[Dict[str, Any]] = []

    # Helper to collect bounding boxes of components filtered by predicate
    def collect_bboxes(comps: List[Any], predicate) -> List[List[List[float]]]:
        boxes: List[List[List[float]]] = []
        for c in comps:
            if predicate(c):
                if c.bbox is not None:
                    boxes.append(c.bbox.tolist())
        return boxes

    # Process training scenes
    for record in train_records:
        for ann in record['annotations']:
            if not ann['task_type'].startswith('missing_detection'):
                continue
            # Append the full conversation for SFT training
            sft_entries.append({
                'id': ann['id'],
                'point': ann['point'],
                'conversations': ann['conversations'],
            })

    # Process validation scenes
    for record in val_records:
        comps = record['components']
        # Build instance_id -> component map for quick lookup
        comp_map = {c.instance_id: c for c in comps}
        # Precompute missing components by type
        missing_comps = [c for c in comps if c.semantic_id == 10]
        missing_verticals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_vertical']
        missing_horizontals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_horizontal']
        missing_platforms = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_platform']
        present_platforms = [c for c in comps if c.semantic_id == 3]
        present_verticals = [c for c in comps if c.semantic_id == 0]
        present_horizontals = [c for c in comps if c.semantic_id == 1]

        for ann in record['annotations']:
            task_type: str = ann.get('task_type', '')
            if not task_type.startswith('missing_detection'):
                continue
            convs = ann.get('conversations', [])
            if not convs or len(convs) < 2:
                continue
            question_text = convs[0]['value'].replace('<point>\n', '').strip()
            eval_questions.append({
                'question_id': ann['id'],
                'point': ann['point'],
                'text': question_text,
                'category': 'scaffold',
            })
            # Determine label
            if (
                'specific_positive' in task_type
                or task_type.endswith('_none')
                or task_type == 'missing_detection_negative'
            ):
                label = 'No'
            else:
                label = 'Yes'
            # Determine bounding boxes for this question
            bboxes: List[List[List[float]]] = []
            if task_type == 'missing_detection_summary':
                # All missing components
                bboxes = [c.bbox.tolist() for c in missing_comps if c.bbox is not None]
            elif task_type == 'missing_detection_floor':
                floor = ann.get('target_floor')
                bboxes = collect_bboxes(missing_comps, lambda c: (c.metadata or {}).get('floor') == floor)
            elif task_type == 'missing_detection_bay':
                bay = ann.get('target_bay')
                bboxes = collect_bboxes(missing_comps, lambda c: (c.metadata or {}).get('bay') == bay)
            elif task_type == 'missing_detection_specific':
                inst_id = ann.get('target_instance_id')
                c = comp_map.get(inst_id)
                if c is not None and c.bbox is not None:
                    bboxes = [c.bbox.tolist()]
            elif task_type == 'missing_detection_specific_positive':
                inst_id = ann.get('target_instance_id')
                c = comp_map.get(inst_id)
                if c is not None and c.bbox is not None:
                    bboxes = [c.bbox.tolist()]
            elif task_type == 'missing_detection_vertical_summary':
                bboxes = [c.bbox.tolist() for c in missing_verticals if c.bbox is not None]
            elif task_type == 'missing_detection_vertical_specific':
                inst_id = ann.get('target_instance_id')
                c = comp_map.get(inst_id)
                if c is not None and c.bbox is not None:
                    bboxes = [c.bbox.tolist()]
            elif task_type == 'missing_detection_vertical_specific_positive':
                inst_id = ann.get('target_instance_id')
                c = comp_map.get(inst_id)
                if c is not None and c.bbox is not None:
                    bboxes = [c.bbox.tolist()]
            elif task_type == 'missing_detection_horizontal_summary':
                bboxes = [c.bbox.tolist() for c in missing_horizontals if c.bbox is not None]
            elif task_type == 'missing_detection_horizontal_specific':
                inst_id = ann.get('target_instance_id')
                c = comp_map.get(inst_id)
                if c is not None and c.bbox is not None:
                    bboxes = [c.bbox.tolist()]
            elif task_type == 'missing_detection_horizontal_specific_positive':
                inst_id = ann.get('target_instance_id')
                c = comp_map.get(inst_id)
                if c is not None and c.bbox is not None:
                    bboxes = [c.bbox.tolist()]
            else:
                # For negative summary cases or unknown types, leave bboxes empty
                bboxes = []
            eval_gt.append({
                'question_id': ann['id'],
                'point': ann['point'],
                'text': label,
                'bboxes': bboxes,
                'category': 'scaffold',
            })

    # Write training SFT file
    sft_path = os.path.join(output_dir, 'scaffold_missing_sft.json')
    with open(sft_path, 'w', encoding='utf-8') as f:
        json.dump(sft_entries, f, indent=2)

    # Write evaluation questions
    q_path = os.path.join(output_dir, 'question_missing.jsonl')
    with open(q_path, 'w', encoding='utf-8') as f:
        for q in eval_questions:
            f.write(json.dumps(q) + '\n')

    # Write evaluation ground truth
    gt_path = os.path.join(output_dir, 'gt_missing.jsonl')
    with open(gt_path, 'w', encoding='utf-8') as f:
        for g in eval_gt:
            f.write(json.dumps(g) + '\n')

    print(f"Generated {len(scene_records)} scenes: {len(train_records)} train and {len(val_records)} val.")
    print(f"SFT entries: {len(sft_entries)}, evaluation questions: {len(eval_questions)}")
    print(f"Files saved in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a missing component detection dataset for scaffolds.')
    parser.add_argument('--output-dir', type=str, default='./playground/data/shapellm/scaffold_missing', help='Directory to save dataset files.')
    parser.add_argument('--num-scenes', type=int, default=1000, help='Number of scenes to generate.')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Fraction of scenes to use for training.')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Fraction of scenes to use for validation.')
    args = parser.parse_args()
    generate_missing_dataset(
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )