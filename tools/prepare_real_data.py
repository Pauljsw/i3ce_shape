#!/usr/bin/env python3
"""
Real Scaffold Point Cloud Preprocessing for Validation

이 스크립트는 실제 비계 점군 데이터를 모델 검증용으로 변환합니다.
- 점군 정규화 (centering, scaling)
- 부재 제거를 통한 결함 점군 생성
- Ground Truth 라벨 및 평가 파일 생성

============================================================================
필요한 수작업 입력 파일 (2개)
============================================================================

1. 원본 점군 파일
   - 위치: --input-pointcloud 로 지정
   - 형식: .ply, .pcd, .las, .xyz, .npy 등
   - 내용: 완전한 비계의 3D 점군

2. 부재 좌표 CSV 파일
   - 위치: --components-csv 로 지정
   - 형식: CSV (콤마 구분)
   - 내용: 각 부재의 bounding box 좌표

   ★ 형식 A: Center/Dimension (CloudCompare에서 바로 복사 - 권장!)
   ┌─────────────────────────────────────────────────────────────────────┐
   │ component_id,type,cx,cy,cz,dx,dy,dz                                │
   │ c1,vertical,-5.937477,5.495575,116.242111,1.31834,0.200557,0.14    │
   │ c2,horizontal,0.5,3.2,112.5,10.5,0.15,0.08                         │
   │ c3,platform,0.0,2.5,115.0,2.0,1.5,0.05                             │
   └─────────────────────────────────────────────────────────────────────┘

   CloudCompare에서 복사할 값:
   - cx, cy, cz: Global Box Center의 X, Y, Z
   - dx, dy, dz: Box Dimensions의 X, Y, Z

   형식 B: Min/Max (직접 계산한 경우)
   ┌─────────────────────────────────────────────────────────────────────┐
   │ component_id,type,x_min,x_max,y_min,y_max,z_min,z_max              │
   │ c1,vertical,-6.60,-5.28,5.40,5.60,116.17,116.31                    │
   └─────────────────────────────────────────────────────────────────────┘

   공통:
   - component_id: 부재 고유 이름 (자유롭게, 예: c1, v1, part_a)
   - type: 부재 종류 (vertical, horizontal, platform 중 하나)

============================================================================
사용법
============================================================================

# 기본 사용
python prepare_real_data.py \
    --input-pointcloud ./scaffold_A.ply \
    --components-csv ./scaffold_A_components.csv \
    --output-dir ./real_validation_data \
    --scaffold-id scaffold_A

# 복합 결함 생성 포함
python prepare_real_data.py \
    --input-pointcloud ./scaffold_A.ply \
    --components-csv ./scaffold_A_components.csv \
    --output-dir ./real_validation_data \
    --scaffold-id scaffold_A \
    --max-combinations 2

============================================================================
출력 파일
============================================================================

output-dir/
├── pointclouds/                    # 정규화된 점군 파일들
│   ├── scaffold_A_complete.npy     # 완전한 비계 (baseline)
│   ├── scaffold_A_c1_removed.npy   # c1 부재 제거
│   ├── scaffold_A_c2_removed.npy   # c2 부재 제거
│   └── ...
├── real_test_questions.jsonl       # 평가용 질문
├── real_test_gt.jsonl              # Ground Truth
└── generation_log.json             # 생성 로그

"""

import os
import sys
import csv
import json
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

# Optional dependencies for various point cloud formats
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Component:
    """부재 정보"""
    component_id: str
    type: str  # vertical, horizontal, platform
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def get_bbox_corners(self) -> np.ndarray:
        """8개 꼭짓점 반환 (모델 출력 형식과 동일)"""
        return np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_min, self.y_max, self.z_max],
            [self.x_max, self.y_max, self.z_max],
        ])

    def contains_point(self, point: np.ndarray) -> bool:
        """점이 bbox 내부에 있는지 확인"""
        return (self.x_min <= point[0] <= self.x_max and
                self.y_min <= point[1] <= self.y_max and
                self.z_min <= point[2] <= self.z_max)

    def transform(self, centroid: np.ndarray, scale: float) -> 'Component':
        """좌표 변환 적용 (centering + scaling)"""
        return Component(
            component_id=self.component_id,
            type=self.type,
            x_min=(self.x_min - centroid[0]) / scale,
            x_max=(self.x_max - centroid[0]) / scale,
            y_min=(self.y_min - centroid[1]) / scale,
            y_max=(self.y_max - centroid[1]) / scale,
            z_min=(self.z_min - centroid[2]) / scale,
            z_max=(self.z_max - centroid[2]) / scale,
        )


@dataclass
class GeneratedSample:
    """생성된 샘플 정보"""
    scene_id: str
    point_file: str
    removed_components: List[Component]
    has_missing: bool
    missing_count: int
    missing_types: Dict[str, int]


# ============================================================================
# Point Cloud I/O
# ============================================================================

def load_pointcloud(filepath: str) -> np.ndarray:
    """
    다양한 형식의 점군 파일 로드

    지원 형식: .npy, .ply, .pcd, .las, .laz, .xyz, .txt
    반환: (N, 3) 또는 (N, 6+) numpy array (XYZ 또는 XYZRGB...)
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    print(f"Loading point cloud: {filepath}")

    if ext == '.npy':
        points = np.load(filepath)

    elif ext in ['.ply', '.pcd']:
        if not HAS_OPEN3D:
            raise ImportError("open3d required for .ply/.pcd files: pip install open3d")
        pcd = o3d.io.read_point_cloud(str(filepath))
        points = np.asarray(pcd.points)

    elif ext in ['.las', '.laz']:
        if not HAS_LASPY:
            raise ImportError("laspy required for .las/.laz files: pip install laspy")
        las = laspy.read(str(filepath))
        points = np.vstack([las.x, las.y, las.z]).T

    elif ext in ['.xyz', '.txt']:
        # Space or comma separated XYZ
        points = np.loadtxt(filepath, delimiter=None)
        if points.ndim == 1:
            points = points.reshape(-1, 3)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Ensure at least XYZ
    if points.shape[1] < 3:
        raise ValueError(f"Point cloud must have at least 3 columns (XYZ), got {points.shape[1]}")

    # Use only XYZ
    points = points[:, :3].astype(np.float32)

    print(f"  Loaded {len(points):,} points")
    print(f"  Bounds: X[{points[:,0].min():.3f}, {points[:,0].max():.3f}], "
          f"Y[{points[:,1].min():.3f}, {points[:,1].max():.3f}], "
          f"Z[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

    return points


def save_pointcloud(points: np.ndarray, filepath: str):
    """점군을 .npy 형식으로 저장"""
    np.save(filepath, points.astype(np.float32))


# ============================================================================
# CSV Loading
# ============================================================================

def load_components_csv(csv_path: str) -> List[Component]:
    """
    부재 좌표 CSV 파일 로드

    지원 형식 2가지:

    형식 A (Min/Max):
        component_id,type,x_min,x_max,y_min,y_max,z_min,z_max
        c1,vertical,-6.59,-5.28,5.39,5.60,116.17,116.31

    형식 B (Center/Dimension) - CloudCompare에서 바로 복사:
        component_id,type,cx,cy,cz,dx,dy,dz
        c1,vertical,-5.937477,5.495575,116.242111,1.31834,0.200557,0.140015

        cx,cy,cz = Global Box Center (X,Y,Z)
        dx,dy,dz = Box Dimensions (X,Y,Z)
    """
    components = []

    print(f"Loading components CSV: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Detect format
        minmax_cols = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
        center_cols = ['cx', 'cy', 'cz', 'dx', 'dy', 'dz']

        use_center_format = all(col in fieldnames for col in center_cols)
        use_minmax_format = all(col in fieldnames for col in minmax_cols)

        if use_center_format:
            print("  Detected format: Center/Dimension (cx,cy,cz,dx,dy,dz)")
        elif use_minmax_format:
            print("  Detected format: Min/Max (x_min,x_max,...)")
        else:
            raise ValueError(
                f"CSV must have either:\n"
                f"  - Min/Max columns: {minmax_cols}\n"
                f"  - Center/Dimension columns: {center_cols}\n"
                f"  Found columns: {fieldnames}"
            )

        for row in reader:
            component_id = row['component_id'].strip()
            comp_type = row['type'].strip().lower()

            if use_center_format:
                # Center/Dimension format: calculate min/max
                cx = float(row['cx'])
                cy = float(row['cy'])
                cz = float(row['cz'])
                dx = float(row['dx'])
                dy = float(row['dy'])
                dz = float(row['dz'])

                x_min = cx - dx / 2
                x_max = cx + dx / 2
                y_min = cy - dy / 2
                y_max = cy + dy / 2
                z_min = cz - dz / 2
                z_max = cz + dz / 2
            else:
                # Min/Max format: use directly
                x_min = float(row['x_min'])
                x_max = float(row['x_max'])
                y_min = float(row['y_min'])
                y_max = float(row['y_max'])
                z_min = float(row['z_min'])
                z_max = float(row['z_max'])

            comp = Component(
                component_id=component_id,
                type=comp_type,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )

            # Validate type
            if comp.type not in ['vertical', 'horizontal', 'platform']:
                print(f"  Warning: Unknown type '{comp.type}' for {comp.component_id}, "
                      f"expected: vertical, horizontal, platform")

            components.append(comp)

    print(f"  Loaded {len(components)} components")

    # Summary by type
    type_counts = {}
    for c in components:
        type_counts[c.type] = type_counts.get(c.type, 0) + 1
    for t, cnt in type_counts.items():
        print(f"    - {t}: {cnt}")

    return components


# ============================================================================
# Point Cloud Processing
# ============================================================================

def normalize_pointcloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    점군 정규화: centering + scaling to [-1, 1]

    Returns:
        normalized_points: 정규화된 점군
        centroid: 원본 중심점 (변환 역산용)
        scale: 스케일 팩터 (변환 역산용)
    """
    # Centering
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Scaling to [-1, 1]
    max_dist = np.abs(centered).max()
    scale = max_dist if max_dist > 0 else 1.0
    normalized = centered / scale

    print(f"Normalization applied:")
    print(f"  Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"  Scale factor: {scale:.3f}")
    print(f"  New bounds: [{normalized.min():.3f}, {normalized.max():.3f}]")

    return normalized.astype(np.float32), centroid, scale


def remove_component(points: np.ndarray, component: Component) -> np.ndarray:
    """
    점군에서 특정 부재(bbox 영역) 제거
    """
    # Create mask for points OUTSIDE the bbox
    mask = ~(
        (points[:, 0] >= component.x_min) & (points[:, 0] <= component.x_max) &
        (points[:, 1] >= component.y_min) & (points[:, 1] <= component.y_max) &
        (points[:, 2] >= component.z_min) & (points[:, 2] <= component.z_max)
    )

    removed_count = (~mask).sum()
    result = points[mask]

    return result, removed_count


def subsample_points(points: np.ndarray, n_points: int = 8192) -> np.ndarray:
    """
    점군 서브샘플링 (모델 입력 크기에 맞춤)
    """
    if len(points) > n_points:
        # Random subsample
        indices = np.random.choice(len(points), n_points, replace=False)
        return points[indices]
    elif len(points) < n_points:
        # Pad with random duplicates
        indices = np.random.choice(len(points), n_points - len(points), replace=True)
        return np.vstack([points, points[indices]])
    return points


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_question_and_gt(
    sample: GeneratedSample,
    scaffold_id: str
) -> Tuple[Dict, Dict]:
    """
    평가용 질문과 GT 생성
    """
    question_id = f"real_{sample.scene_id}"

    # Question (항상 동일한 형식 - data leakage 방지)
    question = {
        "question_id": question_id,
        "point": sample.point_file,
        "text": "This is a scaffold structure. Are there any missing components? "
                "If yes, describe what components are missing and their locations.",
        "category": "scaffold_real"
    }

    # Ground Truth
    if sample.has_missing:
        # Build answer text
        answer_lines = [f"Yes, missing components detected ({sample.missing_count} total):"]
        for comp in sample.removed_components:
            bbox_str = str(comp.get_bbox_corners().tolist())
            if comp.type == 'vertical':
                answer_lines.append(f"- Vertical post: {bbox_str}")
            elif comp.type == 'horizontal':
                answer_lines.append(f"- Horizontal beam: {bbox_str}")
            elif comp.type == 'platform':
                answer_lines.append(f"- Platform: {bbox_str}")

        answer_text = "\n".join(answer_lines)
        label = "Yes"
        bboxes = [comp.get_bbox_corners().tolist() for comp in sample.removed_components]
    else:
        answer_text = "No missing components detected. All scaffold components are properly installed."
        label = "No"
        bboxes = []

    gt = {
        "question_id": question_id,
        "point": sample.point_file,
        "text": answer_text,
        "label": label,
        "bboxes": bboxes,
        "task_type": "missing_detection_summary",
        "missing_count": sample.missing_count,
        "missing_types": sample.missing_types,
        "source": "real_data",
        "scaffold_id": scaffold_id
    }

    return question, gt


def generate_defective_samples(
    points_normalized: np.ndarray,
    components_normalized: List[Component],
    scaffold_id: str,
    output_dir: Path,
    max_combinations: int = 1,
    n_points: int = 8192
) -> List[GeneratedSample]:
    """
    결함 점군 샘플들 생성
    """
    samples = []
    pc_dir = output_dir / "pointclouds"
    pc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Complete scaffold (baseline)
    print("\nGenerating complete scaffold (baseline)...")
    complete_points = subsample_points(points_normalized.copy(), n_points)
    complete_file = f"{scaffold_id}_complete.npy"
    save_pointcloud(complete_points, pc_dir / complete_file)

    samples.append(GeneratedSample(
        scene_id=f"{scaffold_id}_complete",
        point_file=complete_file,
        removed_components=[],
        has_missing=False,
        missing_count=0,
        missing_types={}
    ))

    # 2. Single component removal
    print(f"\nGenerating single-component defects ({len(components_normalized)} variants)...")
    for comp in components_normalized:
        defect_points, removed_count = remove_component(points_normalized.copy(), comp)

        if removed_count == 0:
            print(f"  Warning: No points removed for {comp.component_id}, skipping")
            continue

        defect_points = subsample_points(defect_points, n_points)
        defect_file = f"{scaffold_id}_{comp.component_id}_removed.npy"
        save_pointcloud(defect_points, pc_dir / defect_file)

        samples.append(GeneratedSample(
            scene_id=f"{scaffold_id}_{comp.component_id}_removed",
            point_file=defect_file,
            removed_components=[comp],
            has_missing=True,
            missing_count=1,
            missing_types={comp.type: 1}
        ))

        print(f"  Created: {defect_file} (removed {removed_count:,} points)")

    # 3. Multi-component removal (combinations)
    if max_combinations >= 2 and len(components_normalized) >= 2:
        print(f"\nGenerating multi-component defects (combinations of 2)...")

        combo_count = 0
        max_combos = 20  # Limit to avoid explosion

        for combo in itertools.combinations(components_normalized, 2):
            if combo_count >= max_combos:
                print(f"  Reached max combinations limit ({max_combos})")
                break

            defect_points = points_normalized.copy()
            total_removed = 0

            for comp in combo:
                defect_points, removed = remove_component(defect_points, comp)
                total_removed += removed

            if total_removed == 0:
                continue

            defect_points = subsample_points(defect_points, n_points)
            combo_ids = "_".join([c.component_id for c in combo])
            defect_file = f"{scaffold_id}_{combo_ids}_removed.npy"
            save_pointcloud(defect_points, pc_dir / defect_file)

            type_counts = {}
            for c in combo:
                type_counts[c.type] = type_counts.get(c.type, 0) + 1

            samples.append(GeneratedSample(
                scene_id=f"{scaffold_id}_{combo_ids}_removed",
                point_file=defect_file,
                removed_components=list(combo),
                has_missing=True,
                missing_count=len(combo),
                missing_types=type_counts
            ))

            combo_count += 1
            print(f"  Created: {defect_file}")

    return samples


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Real Scaffold Point Cloud Preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
필요한 입력 파일
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 점군 파일 (--input-pointcloud)
   - 형식: .ply, .pcd, .las, .npy, .xyz
   - 내용: 완전한 비계 스캔

2. 부재 좌표 CSV (--components-csv)
   - 형식: CSV
   - 필수 컬럼: component_id,type,x_min,x_max,y_min,y_max,z_min,z_max

   예시:
   component_id,type,x_min,x_max,y_min,y_max,z_min,z_max
   c1,vertical,-0.52,-0.42,0.01,0.10,0.00,2.45
   c2,horizontal,-0.52,0.51,0.05,0.07,1.20,1.25

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )

    parser.add_argument('--input-pointcloud', type=str, required=True,
                        help='원본 점군 파일 경로 (.ply, .pcd, .las, .npy, .xyz)')
    parser.add_argument('--components-csv', type=str, required=True,
                        help='부재 좌표 CSV 파일 경로')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='출력 디렉토리')
    parser.add_argument('--scaffold-id', type=str, required=True,
                        help='비계 식별자 (출력 파일 이름에 사용)')
    parser.add_argument('--max-combinations', type=int, default=1,
                        help='최대 복합 결함 수 (1=단일만, 2=2개 조합 포함)')
    parser.add_argument('--n-points', type=int, default=8192,
                        help='출력 점군 포인트 수 (기본: 8192)')
    parser.add_argument('--skip-normalize', action='store_true',
                        help='정규화 건너뛰기 (이미 정규화된 경우)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Real Scaffold Data Preprocessing")
    print("=" * 70)

    # 1. Load point cloud
    points = load_pointcloud(args.input_pointcloud)

    # 2. Load components CSV
    components = load_components_csv(args.components_csv)

    # 3. Normalize
    if args.skip_normalize:
        print("\nSkipping normalization (--skip-normalize)")
        points_normalized = points
        centroid = np.zeros(3)
        scale = 1.0
        components_normalized = components
    else:
        print("\nNormalizing point cloud...")
        points_normalized, centroid, scale = normalize_pointcloud(points)

        # Transform component bboxes with same transformation
        print("\nTransforming component bboxes...")
        components_normalized = [c.transform(centroid, scale) for c in components]

    # 4. Generate samples
    print("\n" + "=" * 70)
    print(" Generating Defective Samples")
    print("=" * 70)

    samples = generate_defective_samples(
        points_normalized=points_normalized,
        components_normalized=components_normalized,
        scaffold_id=args.scaffold_id,
        output_dir=output_dir,
        max_combinations=args.max_combinations,
        n_points=args.n_points
    )

    # 5. Generate evaluation files
    print("\n" + "=" * 70)
    print(" Generating Evaluation Files")
    print("=" * 70)

    questions = []
    ground_truths = []

    for sample in samples:
        q, gt = generate_question_and_gt(sample, args.scaffold_id)
        questions.append(q)
        ground_truths.append(gt)

    # Save questions
    questions_path = output_dir / "real_test_questions.jsonl"
    with open(questions_path, 'w') as f:
        for q in questions:
            f.write(json.dumps(q) + '\n')
    print(f"Saved: {questions_path}")

    # Save ground truth
    gt_path = output_dir / "real_test_gt.jsonl"
    with open(gt_path, 'w') as f:
        for gt in ground_truths:
            f.write(json.dumps(gt) + '\n')
    print(f"Saved: {gt_path}")

    # Save generation log
    log = {
        "timestamp": datetime.now().isoformat(),
        "input_pointcloud": str(args.input_pointcloud),
        "components_csv": str(args.components_csv),
        "scaffold_id": args.scaffold_id,
        "normalization": {
            "applied": not args.skip_normalize,
            "centroid": centroid.tolist() if not args.skip_normalize else None,
            "scale": float(scale) if not args.skip_normalize else None
        },
        "num_components": len(components),
        "num_samples": len(samples),
        "samples_summary": {
            "complete": sum(1 for s in samples if not s.has_missing),
            "with_defects": sum(1 for s in samples if s.has_missing),
            "single_defect": sum(1 for s in samples if s.missing_count == 1),
            "multi_defect": sum(1 for s in samples if s.missing_count > 1),
        }
    }

    log_path = output_dir / "generation_log.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_path}")

    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"Total samples generated: {len(samples)}")
    print(f"  - Complete (baseline): {log['samples_summary']['complete']}")
    print(f"  - With defects: {log['samples_summary']['with_defects']}")
    print(f"    - Single defect: {log['samples_summary']['single_defect']}")
    print(f"    - Multi defect: {log['samples_summary']['multi_defect']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\n다음 단계:")
    print(f"  1. pointclouds/ 폴더를 모델 서버로 복사")
    print(f"  2. 모델 inference 실행")
    print(f"  3. 평가 스크립트 실행:")
    print(f"     python tools/evaluate_scaffold_rigorous.py \\")
    print(f"         --predictions <model_output>.jsonl \\")
    print(f"         --ground-truth {gt_path}")


if __name__ == '__main__':
    main()
