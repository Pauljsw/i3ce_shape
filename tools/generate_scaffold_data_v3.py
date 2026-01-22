"""
🏗️ Scaffold Data Generator V3 - Text Shortcut 제거 버전
- 질문에서 구조 정보 제거 (모델이 point cloud를 반드시 분석해야 함)
- 다양한 질문 템플릿 사용
- Negative sample 강화
"""

import numpy as np
import os
import random
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import defaultdict

# Import base generator
from generate_scaffold_data_improved import (
    EnhancedScaffoldGenerator,
    ScaffoldComponent,
    ScaffoldSpecs,
    KoreanScaffoldRegulations
)


class ScaffoldDataGeneratorV3(EnhancedScaffoldGenerator):
    """V3: Text shortcut 제거 버전"""

    def __init__(self, random_seed=42):
        super().__init__(random_seed)

        # 다양한 질문 템플릿 (구조 정보 없음)
        self.question_templates = {
            # Missing Detection - Summary (NO structure info)
            'missing_summary': [
                '<point>\nAnalyze this scaffold structure. Are there any missing components? If so, provide their locations.',
                '<point>\nInspect this scaffold for completeness. Report any missing parts with their positions.',
                '<point>\nCheck this structure for gaps or missing elements. List all deficiencies found.',
                '<point>\nExamine the scaffold. Identify any components that should be present but are missing.',
                '<point>\nScan this scaffold and report any structural incompleteness.',
            ],

            # Missing Detection - Floor level (minimal hint)
            'missing_floor': [
                '<point>\nCheck floor {floor} for any missing components.',
                '<point>\nInspect level {floor} of this scaffold. Are there gaps?',
                '<point>\nAnalyze floor {floor}. Report any missing parts.',
                '<point>\nExamine the {floor}th level. What components are absent?',
            ],

            # Missing Detection - Bay level (minimal hint)
            'missing_bay': [
                '<point>\nCheck bay {bay} for any missing components.',
                '<point>\nInspect bay section {bay}. Are there gaps?',
                '<point>\nAnalyze bay {bay}. Report any missing parts.',
                '<point>\nExamine bay number {bay}. What is missing?',
            ],

            # Missing Detection - Specific position (NO structure info, just location)
            'missing_specific': [
                '<point>\nIs there a platform at floor {floor}, bay {bay}?',
                '<point>\nCheck position (floor {floor}, bay {bay}). Is the platform present?',
                '<point>\nVerify: Does a platform exist at level {floor}, section {bay}?',
                '<point>\nInspect location floor {floor}, bay {bay} for platform presence.',
            ],

            # Missing Detection - Component type summary
            'missing_vertical': [
                '<point>\nAre all vertical posts present in this scaffold?',
                '<point>\nCheck for missing vertical supports.',
                '<point>\nInspect the vertical structural elements. Any missing?',
                '<point>\nAnalyze vertical posts. Report any gaps.',
            ],

            'missing_horizontal': [
                '<point>\nAre all horizontal beams present in this scaffold?',
                '<point>\nCheck for missing horizontal members.',
                '<point>\nInspect the horizontal structural elements. Any missing?',
                '<point>\nAnalyze horizontal beams. Report any gaps.',
            ],

            # Negative samples (when nothing is missing)
            'no_missing': [
                '<point>\nAre there any missing components in this scaffold?',
                '<point>\nInspect this structure. Is anything missing?',
                '<point>\nCheck this scaffold for completeness.',
                '<point>\nAnalyze this scaffold. Are all components present?',
            ],

            # Safety Assessment (NO structure info)
            'safety_summary': [
                '<point>\nProvide a safety assessment of this scaffold structure.',
                '<point>\nEvaluate the structural safety of this scaffold.',
                '<point>\nInspect this scaffold for safety compliance.',
                '<point>\nAnalyze this structure for safety issues.',
            ],

            # Damage Detection (NO structure info)
            'damage_summary': [
                '<point>\nAre there any damaged components in this scaffold?',
                '<point>\nInspect for structural damage.',
                '<point>\nCheck this scaffold for damaged parts.',
                '<point>\nAnalyze this structure for defects or damage.',
            ],

            # Structure Analysis (model must infer from point cloud)
            'structure_size': [
                '<point>\nDescribe the overall size of this scaffold structure.',
                '<point>\nHow many floors and bays does this scaffold have?',
                '<point>\nAnalyze and report the dimensions of this scaffold.',
                '<point>\nWhat is the configuration of this scaffold?',
            ],

            # Component counting (model must count from point cloud)
            'count_components': [
                '<point>\nHow many vertical posts are in this scaffold?',
                '<point>\nCount the horizontal beams in this structure.',
                '<point>\nHow many platforms are installed?',
            ],
        }

    def _select_template(self, category: str, **kwargs) -> str:
        """템플릿 선택 및 포맷"""
        templates = self.question_templates.get(category, ['<point>\nAnalyze this scaffold.'])
        template = random.choice(templates)
        return template.format(**kwargs) if kwargs else template

    def generate_shapellm_annotations_v3(self, scene_id: str, components: List, config: Dict) -> List[Dict]:
        """V3: Text shortcut 제거된 annotation 생성"""
        annotations = []

        # Extract config
        violations = config.get('regulation_violations', [])
        defect_info = config.get('defect_info', {})
        safety_status = config.get('safety_status', 'safe')
        num_floors = config.get('num_floors', 3)
        num_bays = config.get('num_bays', 3)

        # Partition components
        platforms = [c for c in components if c.semantic_id == 3]
        verticals = [c for c in components if c.semantic_id == 0]
        horizontals = [c for c in components if c.semantic_id == 1]
        missing_comps = [c for c in components if c.semantic_id == 10]
        damaged_comps = [c for c in components if c.semantic_id == 9]

        # ================================================================
        # 1️⃣ Missing Detection
        # ================================================================

        if missing_comps:
            # 1-1. Overall summary (NO structure info in question!)
            missing_info = []
            for comp in missing_comps:
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                bbox_str = self._format_bbox(comp.bbox_norm) if comp.bbox_norm is not None else self._format_bbox(comp.bbox)
                missing_info.append(f"- Floor {floor}, Bay {bay}: {bbox_str}")

            annotations.append({
                'id': f"{scene_id}_missing_detection_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('missing_summary')},
                    {'from': 'gpt', 'value': f"Yes, {len(missing_comps)} components are missing:\n" + '\n'.join(missing_info)}
                ],
                'task_type': 'missing_detection_summary',
                'num_defects': len(missing_comps),
                'gt_answer': 'yes',
                'gt_count': len(missing_comps),
                'gt_bboxes': [self._format_bbox(c.bbox_norm if c.bbox_norm is not None else c.bbox) for c in missing_comps]
            })

            # 1-2. Floor-level questions
            floors_with_missing = defaultdict(list)
            for comp in missing_comps:
                floor = (comp.metadata or {}).get('floor', 0)
                floors_with_missing[floor].append(comp)

            # Ask about floors WITH missing components
            for floor, comps_in_floor in floors_with_missing.items():
                floor_info = []
                for comp in comps_in_floor:
                    bay = (comp.metadata or {}).get('bay', '?')
                    bbox_str = self._format_bbox(comp.bbox_norm if comp.bbox_norm is not None else comp.bbox)
                    floor_info.append(f"- Bay {bay}: {bbox_str}")

                annotations.append({
                    'id': f"{scene_id}_missing_detection_floor_{floor}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_floor', floor=floor)},
                        {'from': 'gpt', 'value': f"Yes, {len(comps_in_floor)} components are missing on floor {floor}:\n" + '\n'.join(floor_info)}
                    ],
                    'task_type': 'missing_detection_floor',
                    'target_floor': floor,
                    'num_defects': len(comps_in_floor),
                    'gt_answer': 'yes',
                    'gt_count': len(comps_in_floor),
                    'gt_bboxes': [self._format_bbox(c.bbox_norm if c.bbox_norm is not None else c.bbox) for c in comps_in_floor]
                })

            # Ask about floors WITHOUT missing components (negative samples)
            all_floors = set(range(num_floors))
            missing_floors = set(floors_with_missing.keys())
            intact_floors = all_floors - missing_floors

            for floor in list(intact_floors)[:2]:  # max 2 negative samples per scene
                annotations.append({
                    'id': f"{scene_id}_missing_detection_floor_{floor}_neg",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_floor', floor=floor)},
                        {'from': 'gpt', 'value': f"No, all components on floor {floor} are present."}
                    ],
                    'task_type': 'missing_detection_floor',
                    'target_floor': floor,
                    'num_defects': 0,
                    'gt_answer': 'no',
                    'gt_count': 0,
                    'gt_bboxes': []
                })

            # 1-3. Bay-level questions
            bays_with_missing = defaultdict(list)
            for comp in missing_comps:
                bay = (comp.metadata or {}).get('bay', 0)
                bays_with_missing[bay].append(comp)

            for bay, comps_in_bay in bays_with_missing.items():
                bay_info = []
                for comp in comps_in_bay:
                    floor = (comp.metadata or {}).get('floor', '?')
                    bbox_str = self._format_bbox(comp.bbox_norm if comp.bbox_norm is not None else comp.bbox)
                    bay_info.append(f"- Floor {floor}: {bbox_str}")

                annotations.append({
                    'id': f"{scene_id}_missing_detection_bay_{bay}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_bay', bay=bay)},
                        {'from': 'gpt', 'value': f"Yes, {len(comps_in_bay)} components are missing in bay {bay}:\n" + '\n'.join(bay_info)}
                    ],
                    'task_type': 'missing_detection_bay',
                    'target_bay': bay,
                    'num_defects': len(comps_in_bay),
                    'gt_answer': 'yes',
                    'gt_count': len(comps_in_bay),
                    'gt_bboxes': [self._format_bbox(c.bbox_norm if c.bbox_norm is not None else c.bbox) for c in comps_in_bay]
                })

            # Negative bay samples
            all_bays = set(range(num_bays))
            missing_bays = set(bays_with_missing.keys())
            intact_bays = all_bays - missing_bays

            for bay in list(intact_bays)[:2]:
                annotations.append({
                    'id': f"{scene_id}_missing_detection_bay_{bay}_neg",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_bay', bay=bay)},
                        {'from': 'gpt', 'value': f"No, all components in bay {bay} are present."}
                    ],
                    'task_type': 'missing_detection_bay',
                    'target_bay': bay,
                    'num_defects': 0,
                    'gt_answer': 'no',
                    'gt_count': 0,
                    'gt_bboxes': []
                })

            # 1-4. Specific position questions
            # Positive: missing positions
            for idx, comp in enumerate(missing_comps[:5]):  # max 5
                metadata = comp.metadata or {}
                floor = metadata.get('floor', 0)
                bay = metadata.get('bay', 0)
                bbox_str = self._format_bbox(comp.bbox_norm if comp.bbox_norm is not None else comp.bbox)

                annotations.append({
                    'id': f"{scene_id}_missing_detection_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_specific', floor=floor, bay=bay)},
                        {'from': 'gpt', 'value': f"No, the platform at floor {floor}, bay {bay} is missing. Expected location: {bbox_str}."}
                    ],
                    'task_type': 'missing_detection_specific',
                    'target_floor': floor,
                    'target_bay': bay,
                    'gt_answer': 'no',  # platform is NOT present
                    'gt_bbox': bbox_str
                })

            # Negative: intact positions (where platforms exist)
            platform_positions = set()
            for p in platforms:
                meta = p.metadata or {}
                platform_positions.add((meta.get('floor', 0), meta.get('bay', 0)))

            intact_samples = random.sample(list(platform_positions), min(5, len(platform_positions)))
            for idx, (floor, bay) in enumerate(intact_samples):
                # Find the actual platform
                target_platform = None
                for p in platforms:
                    meta = p.metadata or {}
                    if meta.get('floor') == floor and meta.get('bay') == bay:
                        target_platform = p
                        break

                bbox_str = self._format_bbox(target_platform.bbox_norm if target_platform and target_platform.bbox_norm is not None else (target_platform.bbox if target_platform else None))

                annotations.append({
                    'id': f"{scene_id}_missing_detection_specific_neg_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_specific', floor=floor, bay=bay)},
                        {'from': 'gpt', 'value': f"Yes, the platform at floor {floor}, bay {bay} is present. Location: {bbox_str}."}
                    ],
                    'task_type': 'missing_detection_specific',
                    'target_floor': floor,
                    'target_bay': bay,
                    'gt_answer': 'yes',  # platform IS present
                    'gt_bbox': bbox_str
                })

            # 1-5. Component type summary (vertical/horizontal)
            # Check for missing verticals (we don't actually have vertical missing in current data, but add template)
            annotations.append({
                'id': f"{scene_id}_missing_detection_vertical_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('missing_vertical')},
                    {'from': 'gpt', 'value': f"All {len(verticals)} vertical posts are present."}
                ],
                'task_type': 'missing_detection_vertical_summary',
                'gt_answer': 'no',
                'gt_count': 0
            })

            annotations.append({
                'id': f"{scene_id}_missing_detection_horizontal_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('missing_horizontal')},
                    {'from': 'gpt', 'value': f"All {len(horizontals)} horizontal beams are present."}
                ],
                'task_type': 'missing_detection_horizontal_summary',
                'gt_answer': 'no',
                'gt_count': 0
            })

        else:
            # No missing components - negative samples
            annotations.append({
                'id': f"{scene_id}_missing_detection_summary_neg",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('no_missing')},
                    {'from': 'gpt', 'value': "No, all components are present. The scaffold is complete."}
                ],
                'task_type': 'missing_detection_summary',
                'num_defects': 0,
                'gt_answer': 'no',
                'gt_count': 0,
                'gt_bboxes': []
            })

            # Random floor/bay negative samples
            for floor in random.sample(range(num_floors), min(2, num_floors)):
                annotations.append({
                    'id': f"{scene_id}_missing_detection_floor_{floor}_neg",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_floor', floor=floor)},
                        {'from': 'gpt', 'value': f"No, all components on floor {floor} are present."}
                    ],
                    'task_type': 'missing_detection_floor',
                    'target_floor': floor,
                    'gt_answer': 'no',
                    'gt_count': 0
                })

            for bay in random.sample(range(num_bays), min(2, num_bays)):
                annotations.append({
                    'id': f"{scene_id}_missing_detection_bay_{bay}_neg",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_bay', bay=bay)},
                        {'from': 'gpt', 'value': f"No, all components in bay {bay} are present."}
                    ],
                    'task_type': 'missing_detection_bay',
                    'target_bay': bay,
                    'gt_answer': 'no',
                    'gt_count': 0
                })

            # Specific position negative samples (where platforms exist)
            platform_positions = []
            for p in platforms:
                meta = p.metadata or {}
                platform_positions.append((meta.get('floor', 0), meta.get('bay', 0), p))

            for idx, (floor, bay, platform) in enumerate(random.sample(platform_positions, min(5, len(platform_positions)))):
                bbox_str = self._format_bbox(platform.bbox_norm if platform.bbox_norm is not None else platform.bbox)
                annotations.append({
                    'id': f"{scene_id}_missing_detection_specific_neg_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {'from': 'human', 'value': self._select_template('missing_specific', floor=floor, bay=bay)},
                        {'from': 'gpt', 'value': f"Yes, the platform at floor {floor}, bay {bay} is present. Location: {bbox_str}."}
                    ],
                    'task_type': 'missing_detection_specific',
                    'target_floor': floor,
                    'target_bay': bay,
                    'gt_answer': 'yes',
                    'gt_bbox': bbox_str
                })

        # ================================================================
        # 2️⃣ Structure Analysis (model must infer from point cloud!)
        # ================================================================

        annotations.append({
            'id': f"{scene_id}_structure_size",
            'point': f"{scene_id}.npy",
            'conversations': [
                {'from': 'human', 'value': self._select_template('structure_size')},
                {'from': 'gpt', 'value': f"This scaffold has {num_bays} bays and {num_floors} floors."}
            ],
            'task_type': 'structure_analysis',
            'gt_num_bays': num_bays,
            'gt_num_floors': num_floors
        })

        # Component counting
        annotations.append({
            'id': f"{scene_id}_count_verticals",
            'point': f"{scene_id}.npy",
            'conversations': [
                {'from': 'human', 'value': '<point>\nHow many vertical posts are in this scaffold?'},
                {'from': 'gpt', 'value': f"There are {len(verticals)} vertical posts in this scaffold."}
            ],
            'task_type': 'component_counting',
            'component_type': 'vertical',
            'gt_count': len(verticals)
        })

        annotations.append({
            'id': f"{scene_id}_count_horizontals",
            'point': f"{scene_id}.npy",
            'conversations': [
                {'from': 'human', 'value': '<point>\nHow many horizontal beams are in this scaffold?'},
                {'from': 'gpt', 'value': f"There are {len(horizontals)} horizontal beams in this scaffold."}
            ],
            'task_type': 'component_counting',
            'component_type': 'horizontal',
            'gt_count': len(horizontals)
        })

        annotations.append({
            'id': f"{scene_id}_count_platforms",
            'point': f"{scene_id}.npy",
            'conversations': [
                {'from': 'human', 'value': '<point>\nHow many platforms are installed in this scaffold?'},
                {'from': 'gpt', 'value': f"There are {len(platforms)} platforms installed in this scaffold."}
            ],
            'task_type': 'component_counting',
            'component_type': 'platform',
            'gt_count': len(platforms)
        })

        # ================================================================
        # 3️⃣ Safety Assessment (NO structure info)
        # ================================================================

        if violations:
            translated = [self._translate_violation(v) for v in violations]
            annotations.append({
                'id': f"{scene_id}_safety_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('safety_summary')},
                    {'from': 'gpt', 'value': f"This scaffold has {len(violations)} safety issues:\n" + '\n'.join([f'- {tv}' for tv in translated[:5]])}
                ],
                'task_type': 'safety_assessment',
                'safety_status': safety_status,
                'gt_answer': 'unsafe',
                'num_violations': len(violations)
            })
        else:
            annotations.append({
                'id': f"{scene_id}_safety_summary_pass",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('safety_summary')},
                    {'from': 'gpt', 'value': "This scaffold is structurally safe. No safety issues detected."}
                ],
                'task_type': 'safety_assessment',
                'safety_status': 'safe',
                'gt_answer': 'safe',
                'num_violations': 0
            })

        # ================================================================
        # 4️⃣ Damage Detection (NO structure info)
        # ================================================================

        if damaged_comps:
            damage_info = []
            for comp in damaged_comps:
                metadata = comp.metadata or {}
                defect_type = metadata.get('defect_type', 'unknown')
                defect_en = self.defect_en_map.get(defect_type, defect_type)
                bbox_str = self._format_bbox(comp.bbox_norm if comp.bbox_norm is not None else comp.bbox)
                damage_info.append(f"- {defect_en}: {bbox_str}")

            annotations.append({
                'id': f"{scene_id}_damage_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('damage_summary')},
                    {'from': 'gpt', 'value': f"Yes, {len(damaged_comps)} damaged components found:\n" + '\n'.join(damage_info)}
                ],
                'task_type': 'damage_detection',
                'gt_answer': 'yes',
                'num_damaged': len(damaged_comps)
            })
        else:
            annotations.append({
                'id': f"{scene_id}_damage_summary_neg",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {'from': 'human', 'value': self._select_template('damage_summary')},
                    {'from': 'gpt', 'value': "No, all components are in good condition. No damage detected."}
                ],
                'task_type': 'damage_detection',
                'gt_answer': 'no',
                'num_damaged': 0
            })

        return annotations

    def generate_scene_data(self, scene_id):
        """Override: V3 annotation 사용"""
        # 기존 scene 생성 로직 사용
        scene_data = super().generate_scene_data(scene_id)

        if scene_data is None:
            return None

        # V3 annotation으로 교체
        scene_data['annotations'] = self.generate_shapellm_annotations_v3(
            scene_id,
            scene_data['components'],
            scene_data['config']
        )

        return scene_data

    def save_for_shapellm(self, output_dir, num_scenes=2000, train_ratio=0.8, val_ratio=0.1):
        """ShapeLLM 형식으로 저장 (V3: 더 많은 데이터)"""
        output_path = Path(output_dir)
        pcs_dir = output_path / 'pcs'
        meta_dir = output_path / 'meta'
        labels_dir = output_path / 'labels'

        for d in [pcs_dir, meta_dir, labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        all_annotations = []
        all_scene_ids = []
        stats = defaultdict(int)
        task_stats = defaultdict(int)

        print(f"🏗️ ShapeLLM V3 데이터 생성 시작 ({num_scenes} scenes)...")
        print("📝 V3 특징: Text shortcut 제거, 다양한 질문 템플릿, Negative sample 강화")

        for i in range(num_scenes):
            scene_id = f"scaffold_{i:05d}"

            scene_data = self.generate_scene_data(scene_id)
            if scene_data is None:
                print(f"⚠️ Failed: {scene_id}")
                continue

            # .npy 파일 저장
            np.save(pcs_dir / f"{scene_id}.npy", scene_data['coord'].astype(np.float32))

            # Meta 저장
            scene_meta = {
                'scene_id': scene_id,
                'config': scene_data['config'],
                'norm_params': scene_data['norm_params']
            }
            with open(meta_dir / f"{scene_id}_meta.json", 'w', encoding='utf-8') as f:
                json.dump(scene_meta, f, indent=2, ensure_ascii=False)

            # Labels 저장
            labels = []
            for comp in scene_data['components']:
                label = {
                    'instance_id': comp.instance_id,
                    'name': comp.name,
                    'class': self.class_names[comp.semantic_id],
                    'semantic_id': comp.semantic_id,
                    'bbox_world': comp.bbox.tolist() if comp.bbox is not None else None,
                    'bbox_norm': comp.bbox_norm.tolist() if comp.bbox_norm is not None else None,
                    'metadata': comp.metadata
                }
                labels.append(label)

            with open(labels_dir / f"{scene_id}_label.json", 'w', encoding='utf-8') as f:
                json.dump(labels, f, indent=2, ensure_ascii=False)

            # Annotations
            all_annotations.extend(scene_data['annotations'])
            all_scene_ids.append(scene_id)

            # Task 통계
            for ann in scene_data['annotations']:
                task_stats[ann['task_type']] += 1

            stats['total'] += 1
            stats[scene_data['config']['safety_status']] += 1

            if (i + 1) % 200 == 0:
                print(f"  진행: {i + 1}/{num_scenes}")

        # Train/val/test split
        n = len(all_scene_ids)
        indices = np.arange(n)
        np.random.shuffle(indices)

        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        split = {
            'train': [all_scene_ids[i] for i in indices[:n_train]],
            'val': [all_scene_ids[i] for i in indices[n_train:n_train+n_val]],
            'test': [all_scene_ids[i] for i in indices[n_train+n_val:]]
        }

        with open(output_path / 'split.json', 'w', encoding='utf-8') as f:
            json.dump(split, f, indent=2, ensure_ascii=False)

        # Scene → annotations 매핑
        scene_to_annotations = defaultdict(list)
        for ann in all_annotations:
            scene_id = ann['point'].replace('.npy', '')
            scene_to_annotations[scene_id].append(ann)

        # Split별 저장
        for split_name in ['train', 'val', 'test']:
            split_annotations = []
            for scene_id in split[split_name]:
                split_annotations.extend(scene_to_annotations[scene_id])

            # JSON 저장
            with open(output_path / f'instructions_{split_name}.json', 'w', encoding='utf-8') as f:
                json.dump(split_annotations, f, indent=2, ensure_ascii=False)

            # Stage 2 SFT 형식으로도 저장
            with open(output_path / f'stage2_sft_{split_name}.json', 'w', encoding='utf-8') as f:
                json.dump(split_annotations, f, indent=2, ensure_ascii=False)

        # GT 파일 생성 (평가용)
        for split_name in ['val', 'test']:
            gt_data = []
            for scene_id in split[split_name]:
                for ann in scene_to_annotations[scene_id]:
                    gt_entry = {
                        'question_id': ann['id'],
                        'point': ann['point'],
                        'task_type': ann['task_type'],
                        'gt_answer': ann.get('gt_answer', ''),
                        'gt_count': ann.get('gt_count', ann.get('num_defects', 0)),
                        'gt_bboxes': ann.get('gt_bboxes', []),
                        'answer': ann['conversations'][1]['value']
                    }
                    gt_data.append(gt_entry)

            with open(output_path / f'{split_name}_gt.jsonl', 'w', encoding='utf-8') as f:
                for entry in gt_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # 메타데이터 저장
        metadata = {
            'version': 'v3',
            'num_scenes': stats['total'],
            'class_names': self.class_names,
            'safety_distribution': {
                'safe': stats['safe'],
                'minor_defect': stats['minor_defect'],
                'major_defect': stats['major_defect']
            },
            'total_annotations': len(all_annotations),
            'task_distribution': dict(task_stats),
            'split': {
                'train': len(split['train']),
                'val': len(split['val']),
                'test': len(split['test'])
            },
            'features': [
                'no_text_shortcuts',
                'diverse_question_templates',
                'enhanced_negative_samples',
                'structure_analysis_questions',
                'component_counting_questions'
            ]
        }

        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n" + "="*60)
        print("✅ ShapeLLM V3 데이터셋 생성 완료!")
        print("="*60)
        print(f"📁 Point Clouds: {pcs_dir} ({stats['total']}개)")
        print(f"📁 Meta: {meta_dir} ({stats['total']}개)")
        print(f"📁 Labels: {labels_dir} ({stats['total']}개)")
        print(f"📄 Split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
        print(f"\n📊 Task 분포:")
        for task, count in sorted(task_stats.items()):
            print(f"  {task}: {count}")
        print(f"\n📊 안전 상태 분포:")
        print(f"  ✅ 안전: {stats['safe']}개")
        print(f"  ⚠️ 경미: {stats['minor_defect']}개")
        print(f"  🚨 심각: {stats['major_defect']}개")

        print("\n🎯 V3 주요 개선사항:")
        print("  ❌ 질문에서 구조 정보 제거 (text shortcut 차단)")
        print("  ✅ 다양한 질문 템플릿 (5+ per task)")
        print("  ✅ Negative sample 강화 (floor/bay/specific)")
        print("  ✅ Structure analysis 질문 추가 (모델이 추론해야 함)")
        print("  ✅ Component counting 질문 추가")

        return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🏗️ ShapeLLM V3 데이터 생성 (Text Shortcut 제거)')
    parser.add_argument('--num_scenes', type=int, default=2000, help='생성할 scene 개수 (기본: 2000)')
    parser.add_argument('--output_dir', type=str, default='./playground/data/shapellm/scaffold_v3',
                        help='출력 디렉토리 경로')
    parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split 비율')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split 비율')
    args = parser.parse_args()

    generator = ScaffoldDataGeneratorV3(random_seed=args.random_seed)
    stats = generator.save_for_shapellm(
        args.output_dir,
        num_scenes=args.num_scenes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
