"""
🏗️ 고도화된 비계 합성 데이터 생성 도구 (ShapeLLM용)
- 한국 산업안전보건기준 준수 (2025년 기준)
- 실제 시스템비계 규격 반영
- 점진적 학습 목표 지원 (Referring → 누락 감지 → 안정성 → 손상 → 규정)
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

@dataclass
class ScaffoldComponent:
    """비계 부품 정의"""
    name: str
    semantic_id: int
    instance_id: int
    points: np.ndarray  # [N, 3] coordinates (color 제거)
    bbox: Optional[np.ndarray] = None  # [8, 3] bounding box corners (world coords)
    bbox_norm: Optional[np.ndarray] = None  # [8, 3] bounding box corners (normalized coords)
    metadata: Optional[Dict] = None  # 추가 메타데이터

class KoreanScaffoldRegulations:
    """한국 산업안전보건기준 (2025년)"""

    # 기둥 간격 기준
    MAX_COLUMN_SPACING_LEDGER = 1.85  # 띠장 방향 (m)
    MAX_COLUMN_SPACING_PURLIN = 1.5   # 장선 방향 (m)

    # 작업발판 기준
    MIN_PLATFORM_WIDTH = 0.40  # 40cm
    MAX_PLATFORM_GAP = 0.03    # 3cm

    # 안전난간 기준
    TOP_RAIL_HEIGHT_MIN = 0.90   # 90cm
    TOP_RAIL_HEIGHT_MAX = 1.20   # 120cm
    MID_RAIL_REQUIRED = True
    TOE_BOARD_MIN_HEIGHT = 0.10  # 10cm

    # 가새 설치 기준
    MAX_BRACE_VERTICAL_SPAN = 5  # 5단 이내
    BRACE_ANGLE_MIN = 40  # 40도
    BRACE_ANGLE_MAX = 60  # 60도

    # 벽 연결재 기준
    MAX_WALL_TIE_SPACING = 5.0  # 수직/수평 5m 이내

    @classmethod
    def check_column_spacing(cls, spacing_x, spacing_y):
        """기둥 간격 검증"""
        violations = []
        if spacing_x > cls.MAX_COLUMN_SPACING_LEDGER:
            violations.append(f"띠장 방향 기둥 간격 초과: {spacing_x:.2f}m > {cls.MAX_COLUMN_SPACING_LEDGER}m")
        if spacing_y > cls.MAX_COLUMN_SPACING_PURLIN:
            violations.append(f"장선 방향 기둥 간격 초과: {spacing_y:.2f}m > {cls.MAX_COLUMN_SPACING_PURLIN}m")
        return violations

    @classmethod
    def check_platform_width(cls, width):
        """발판 폭 검증"""
        if width < cls.MIN_PLATFORM_WIDTH:
            return [f"작업발판 폭 부족: {width:.2f}m < {cls.MIN_PLATFORM_WIDTH}m"]
        return []

class ScaffoldSpecs:
    """비계 부품 규격 (mm 단위를 m로 변환)"""

    # 수직재 (Vertical Posts) - Ø48.6 * 2.3T
    VERTICAL_LENGTHS = {
        'V-38': 3.8,
        'V-19': 1.9,
        'V-09': 0.95,
        'V-04': 0.475
    }

    # 수평재 (Horizontal Beams) - Ø42.7 * 2.3T
    HORIZONTAL_SPECS = {
        'H-18': {'length': 1.768, 'spacing': 1.817},
        'H-15': {'length': 1.463, 'spacing': 1.512},
        'H-12': {'length': 1.158, 'spacing': 1.207},
        'H-09': {'length': 0.853, 'spacing': 0.902},
        'H-06': {'length': 0.549, 'spacing': 0.598},
        'H-03': {'length': 0.244, 'spacing': 0.293}
    }

    # 대각재 (Diagonal Braces) - Ø34 x 2.3T
    DIAGONAL_SPECS = {
        'B-1918': {'length': 2.629, 'height': 1.9, 'width': 1.829},
        'B-1915': {'length': 2.428, 'height': 1.9, 'width': 1.524},
        'B-1912': {'length': 2.251, 'height': 1.9, 'width': 1.219}
    }

    # 발판 (Platform) 크기
    PLATFORM_SIZES = [
        (0.4, 0.598),
        (0.4, 0.902),
        (0.4, 1.817)
    ]

    # 하부받침 (Base Support)
    BASE_SUPPORT = {
        'base_size': (0.14, 0.14),
        'pipe_diameter': 0.034,
        'height': 0.15
    }

    # 파이프 직경
    PIPE_DIAMETERS = {
        'vertical': 0.0486,
        'horizontal': 0.0427,
        'diagonal': 0.034,
        'handrail': 0.034
    }

class EnhancedScaffoldGenerator:
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)

        # 클래스 정의 (의미적 라벨링)
        self.class_names = [
            "수직재",       # 0 - Vertical Posts
            "수평재",       # 1 - Horizontal Beams
            "대각재",       # 2 - Diagonal Braces
            "발판",         # 3 - Platforms
            "하부받침",     # 4 - Base Supports
            "연결부",       # 5 - Connections
            "계단",         # 6 - Stairs
            "사다리",       # 7 - Ladders
            "안전난간",     # 8 - Safety Rails
            "손상부품",     # 9 - Damaged Components
            "누락부분"      # 10 - Missing Parts
        ]

        self.instance_counter = 1
        self.components_registry = []  # 모든 부품 등록

        # English class names corresponding to semantic ids.
        # 0: vertical post, 1: horizontal beam, 2: diagonal brace, 3: platform,
        # 4: base support, 5: connection, 6: stair, 7: ladder,
        # 8: safety rail, 9: damaged component, 10: missing part
        self.class_names_en = [
            "vertical post",     # 0
            "horizontal beam",   # 1
            "diagonal brace",    # 2
            "platform",          # 3
            "base support",      # 4
            "connection",        # 5
            "stair",             # 6
            "ladder",            # 7
            "safety rail",       # 8
            "damaged component", # 9
            "missing part"       # 10
        ]

        # English defect names for damage detection
        self.defect_en_map = {
            'bent': 'bent deformation',
            'cracked': 'crack',
            'corroded': 'corrosion',
            'loose_connection': 'loose connection'
        }

    def _translate_violation(self, violation: str) -> str:
        """Translate Korean regulation violation messages into English.

        The violation strings produced by the regulation checks contain Korean
        phrases such as "띠장 방향 기둥 간격 초과" or "가새 미설치". This helper
        maps common patterns to succinct English descriptions. If no known
        pattern is found, the original string is returned as is.

        Args:
            violation: A violation message in Korean.

        Returns:
            An English translation of the violation.
        """
        # Column spacing exceeded in ledger direction (띠장)
        if '띠장 방향 기둥 간격 초과' in violation:
            # Preserve the numeric values after the colon for context.
            # Example: "띠장 방향 기둥 간격 초과: 1.90m > 1.85m" →
            # "Column spacing exceeded (ledger direction): 1.90m > 1.85m"
            parts = violation.split(':', 1)
            suffix = parts[1].strip() if len(parts) > 1 else ''
            return f'Column spacing exceeded (ledger direction): {suffix}'
        # Column spacing exceeded in purlin direction (장선)
        if '장선 방향 기둥 간격 초과' in violation:
            parts = violation.split(':', 1)
            suffix = parts[1].strip() if len(parts) > 1 else ''
            return f'Column spacing exceeded (purlin direction): {suffix}'
        # Platform width insufficient
        if '작업발판 폭 부족' in violation or '발판 폭 부족' in violation:
            parts = violation.split(':', 1)
            suffix = parts[1].strip() if len(parts) > 1 else ''
            return f'Platform width insufficient: {suffix}'
        # Brace not installed or missing braces
        if '가새' in violation and ('미설치' in violation or '없음' in violation):
            return 'Braces not installed'
        # Safety rail not installed
        if '난간' in violation and ('미설치' in violation or '없음' in violation):
            return 'Safety rail not installed'
        # Generic missing platform
        if '발판 누락' in violation:
            return 'Platform missing'
        # Return the original if no pattern matches
        return violation

    def calculate_bbox(self, points):
        """포인트로부터 8개 꼭짓점 bbox 계산"""
        if len(points) == 0:
            return None

        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)

        # 8개 꼭짓점
        bbox = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
        ])

        return bbox

    def generate_pipe_points(self, start_pos, end_pos, diameter, points_density=100):
        """파이프 형태의 포인트 생성 (색상 제거)"""
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return np.array([]).reshape(0, 3)

        direction = direction / length
        num_points = max(int(length * points_density), 10)
        num_points = min(num_points, 200)

        t_vals = np.linspace(0, 1, num_points)
        points = []

        for t in t_vals:
            center = start_pos + t * (end_pos - start_pos)

            angles = [0, np.pi/2, np.pi, 3*np.pi/2]

            if abs(direction[2]) < 0.9:
                perpendicular = np.cross(direction, [0, 0, 1])
            else:
                perpendicular = np.cross(direction, [1, 0, 0])

            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                perpendicular2 = np.cross(direction, perpendicular)

                radius = diameter / 2

                for angle in angles:
                    offset = radius * (np.cos(angle) * perpendicular + np.sin(angle) * perpendicular2)
                    points.append(center + offset)
            else:
                points.append(center)

        return np.array(points) if points else np.array([]).reshape(0, 3)

    def generate_platform_points(self, center, width, length, points_density=200):
        """발판 포인트 생성 (색상 제거)"""
        points = []

        num_w = max(int(width * points_density), 5)
        num_l = max(int(length * points_density), 5)
        num_w = min(num_w, 20)
        num_l = min(num_l, 20)

        for i in range(num_w):
            for j in range(num_l):
                x = center[0] + (i - num_w/2) * (width / num_w)
                y = center[1] + (j - num_l/2) * (length / num_l)
                z = center[2]

                noise = np.random.normal(0, 0.005, 3)
                points.append([x, y, z] + noise)

        thickness = 0.05
        edge_points = min(num_w, 10)

        for i in range(0, num_w, max(1, num_w//edge_points)):
            for k in [-thickness/2, thickness/2]:
                x = center[0] + (i - num_w/2) * (width / num_w)
                points.append([x, center[1] - length/2, center[2] + k])
                points.append([x, center[1] + length/2, center[2] + k])

        return np.array(points)

    def generate_safety_handrail(self, start_pos, end_pos, height_offset, rail_type='top'):
        """🆕 안전난간 생성"""
        # 난간 위치 계산
        rail_start = start_pos.copy()
        rail_end = end_pos.copy()
        rail_start[2] += height_offset
        rail_end[2] += height_offset

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['handrail']
        points = self.generate_pipe_points(rail_start, rail_end, diameter, 50)

        if len(points) > 0:
            bbox = self.calculate_bbox(points)

            component = ScaffoldComponent(
                name=f"안전난간_{rail_type}_{self.instance_counter}",
                semantic_id=8,  # 안전난간
                instance_id=self.instance_counter,
                points=points,
                bbox=bbox,
                metadata={'type': rail_type, 'height_offset': height_offset}
            )

            self.instance_counter += 1
            return component

        return None

    def create_scaffold_with_regulations(self, config):
        """🆕 규정 준수 검증을 포함한 비계 생성"""
        components = []
        violations = []  # 규정 위반 사항

        # 기본 구조 생성 (기존 로직 활용)
        num_floors = config.get('num_floors', random.randint(3, 5))
        num_bays = config.get('num_bays', random.randint(3, 6))

        # Grid 크기 결정 (실제 규격 기반)
        horizontal_types = list(ScaffoldSpecs.HORIZONTAL_SPECS.keys())
        x_beam_type = random.choice(horizontal_types)
        y_beam_type = random.choice(horizontal_types)

        bay_width = ScaffoldSpecs.HORIZONTAL_SPECS[x_beam_type]['spacing']
        scaffold_depth = ScaffoldSpecs.HORIZONTAL_SPECS[y_beam_type]['spacing']

        # 🆕 규정 검증: 기둥 간격
        spacing_violations = KoreanScaffoldRegulations.check_column_spacing(bay_width, scaffold_depth)
        violations.extend(spacing_violations)

        # 층 높이 설정
        floor_heights = self.generate_diverse_floor_heights(num_floors)
        cumulative_heights = self.get_cumulative_heights(floor_heights)

        config.update({
            'num_floors': num_floors,
            'num_bays': num_bays,
            'bay_width': bay_width,
            'depth': scaffold_depth,
            'floor_heights': floor_heights,
            'cumulative_heights': cumulative_heights,
            'x_beam_type': x_beam_type,
            'y_beam_type': y_beam_type
        })

        # 1. 하부받침
        base_supports = self._create_base_supports_simple(num_bays, bay_width, scaffold_depth)
        components.extend(base_supports)

        # 2. 수직재
        posts = self._create_vertical_posts_simple(num_bays, bay_width, scaffold_depth, cumulative_heights)
        components.extend(posts)

        # 3. 수평재
        beams = self._create_horizontal_beams_simple(num_bays, bay_width, scaffold_depth, cumulative_heights)
        components.extend(beams)

        # 4. 대각재 (가새 규정 검증)
        diagonals, brace_violations = self._create_diagonal_braces_with_validation(
            num_bays, bay_width, scaffold_depth, cumulative_heights, num_floors)
        components.extend(diagonals)
        violations.extend(brace_violations)

        # 5. 발판 (규정 검증)
        platforms, platform_violations = self._create_platforms_with_validation(
            num_bays, bay_width, scaffold_depth, cumulative_heights, config)
        components.extend(platforms)
        violations.extend(platform_violations)

        # 6. 🆕 안전난간 (규정 기반)
        handrails, handrail_violations = self._create_safety_handrails(
            num_bays, bay_width, scaffold_depth, cumulative_heights, config)
        components.extend(handrails)
        violations.extend(handrail_violations)

        # 7. 접근 구조 (계단/사다리)
        access = self._create_access_structures_simple(num_bays, bay_width, scaffold_depth, cumulative_heights)
        components.extend(access)

        # 8. 🆕 결함 적용 (다양한 유형)
        safety_status = config.get('safety_status', self._determine_safety_status())
        config['safety_status'] = safety_status

        if safety_status != 'safe':
            components, defect_info = self._apply_diverse_defects(components, safety_status, config)
            violations.extend(defect_info.get('violations', []))
            config['defect_info'] = defect_info

        # 규정 위반 정보 저장
        config['regulation_violations'] = violations
        config['compliant'] = len(violations) == 0

        # 컴포넌트 등록
        self.components_registry = components

        return components, config

    def generate_diverse_floor_heights(self, num_floors, base_height_range=(1.8, 2.2)):
        """층별 높이 다양화"""
        if random.random() < 0.7:
            uniform_height = random.uniform(*base_height_range)
            return [uniform_height] * num_floors
        else:
            floor_heights = []
            for floor in range(num_floors):
                if floor == 0:
                    height = random.uniform(2.0, 2.5)
                else:
                    height = random.uniform(1.5, 2.3)
                floor_heights.append(height)
            return floor_heights

    def get_cumulative_heights(self, floor_heights):
        """누적 높이 계산"""
        cumulative = [0.1]
        for height in floor_heights:
            cumulative.append(cumulative[-1] + height)
        return cumulative

    def _determine_safety_status(self):
        """안전 상태 결정"""
        prob = random.random()
        if prob < 0.6:  # 60% 안전
            return 'safe'
        elif prob < 0.85:  # 25% 경미한 결함
            return 'minor_defect'
        else:  # 15% 심각한 결함
            return 'major_defect'

    # 간소화된 부품 생성 함수들 (기존 로직 재사용, 색상만 제거)
    def _create_base_supports_simple(self, num_bays, bay_width, depth):
        """하부받침 생성"""
        components = []
        base_spec = ScaffoldSpecs.BASE_SUPPORT

        positions = []
        for i in range(num_bays + 1):
            for j in [0, depth]:
                positions.append([i * bay_width, j, -0.05])

        for pos in positions:
            base_center = np.array(pos)
            base_points = self._generate_base_support_points(base_center, base_spec['base_size'], base_spec['height'])

            if len(base_points) > 0:
                bbox = self.calculate_bbox(base_points)

                component = ScaffoldComponent(
                    name=f"하부받침_{self.instance_counter}",
                    semantic_id=4,
                    instance_id=self.instance_counter,
                    points=base_points,
                    bbox=bbox
                )
                components.append(component)
                self.instance_counter += 1

        return components

    def _generate_base_support_points(self, center, base_size, height):
        """하부받침 포인트 생성"""
        points = []
        width, depth = base_size

        density = 15
        for i in range(density):
            for j in range(density):
                x = center[0] + (i - density/2) * (width / density)
                y = center[1] + (j - density/2) * (depth / density)

                points.append([x, y, center[2] + height])
                if random.random() < 0.3:
                    points.append([x, y, center[2]])

        edge_density = 10
        for i in range(edge_density):
            t = i / edge_density
            edges = [
                [center[0] - width/2 + t*width, center[1] - depth/2, center[2] + height/2],
                [center[0] - width/2 + t*width, center[1] + depth/2, center[2] + height/2],
                [center[0] - width/2, center[1] - depth/2 + t*depth, center[2] + height/2],
                [center[0] + width/2, center[1] - depth/2 + t*depth, center[2] + height/2]
            ]
            points.extend(edges)

        return np.array(points)

    def _create_vertical_posts_simple(self, num_bays, bay_width, depth, cumulative_heights):
        """수직재 생성"""
        components = []
        total_height = cumulative_heights[-1]

        positions = []
        for i in range(num_bays + 1):
            for j in [0, depth]:
                positions.append([i * bay_width, j, 0.1])

        for pos in positions:
            start_pos = np.array(pos)
            end_pos = start_pos + np.array([0, 0, total_height - 0.1])

            diameter = ScaffoldSpecs.PIPE_DIAMETERS['vertical']
            points = self.generate_pipe_points(start_pos, end_pos, diameter)

            if len(points) > 0:
                bbox = self.calculate_bbox(points)

                component = ScaffoldComponent(
                    name=f"수직재_{self.instance_counter}",
                    semantic_id=0,
                    instance_id=self.instance_counter,
                    points=points,
                    bbox=bbox
                )
                components.append(component)
                self.instance_counter += 1

        return components

    def _create_horizontal_beams_simple(self, num_bays, bay_width, depth, cumulative_heights):
        """수평재 생성"""
        components = []

        for z in cumulative_heights[:-1]:
            # X방향
            for bay in range(num_bays):
                for j in [0, depth]:
                    start_pos = np.array([bay * bay_width, j, z])
                    end_pos = np.array([(bay + 1) * bay_width, j, z])

                    diameter = ScaffoldSpecs.PIPE_DIAMETERS['horizontal']
                    points = self.generate_pipe_points(start_pos, end_pos, diameter)

                    if len(points) > 0:
                        bbox = self.calculate_bbox(points)

                        component = ScaffoldComponent(
                            name=f"수평재_X_{self.instance_counter}",
                            semantic_id=1,
                            instance_id=self.instance_counter,
                            points=points,
                            bbox=bbox
                        )
                        components.append(component)
                        self.instance_counter += 1

            # Y방향
            for i in range(num_bays + 1):
                start_pos = np.array([i * bay_width, 0, z])
                end_pos = np.array([i * bay_width, depth, z])

                diameter = ScaffoldSpecs.PIPE_DIAMETERS['horizontal']
                points = self.generate_pipe_points(start_pos, end_pos, diameter)

                if len(points) > 0:
                    bbox = self.calculate_bbox(points)

                    component = ScaffoldComponent(
                        name=f"수평재_Y_{self.instance_counter}",
                        semantic_id=1,
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox
                    )
                    components.append(component)
                    self.instance_counter += 1

        return components

    def _create_diagonal_braces_with_validation(self, num_bays, bay_width, depth, cumulative_heights, num_floors):
        """🆕 대각재 생성 + 규정 검증"""
        components = []
        violations = []

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['diagonal']

        # 가새 설치 패턴 (5단 이내마다 설치해야 함)
        floors_without_braces = []

        for floor_idx in range(len(cumulative_heights) - 1):
            z_bottom = cumulative_heights[floor_idx]
            z_top = cumulative_heights[floor_idx + 1]

            # 60% 확률로 가새 설치 (규정 위반 시뮬레이션)
            if random.random() < 0.6:
                # 앞뒤 가장자리에 대각재 설치
                for j in [0, depth]:
                    for bay in range(0, num_bays, 2):  # 2bay마다
                        start_pos = np.array([bay * bay_width, j, z_bottom])
                        end_pos = np.array([(bay + 1) * bay_width, j, z_top])

                        points = self.generate_pipe_points(start_pos, end_pos, diameter)

                        if len(points) > 0:
                            bbox = self.calculate_bbox(points)

                            component = ScaffoldComponent(
                                name=f"대각재_{floor_idx}층_{self.instance_counter}",
                                semantic_id=2,
                                instance_id=self.instance_counter,
                                points=points,
                                bbox=bbox,
                                metadata={'floor': floor_idx}
                            )
                            components.append(component)
                            self.instance_counter += 1
            else:
                floors_without_braces.append(floor_idx + 1)  # 1층부터 카운트

        # 🆕 규정 검증: 5단 이내 가새 미설치
        if len(floors_without_braces) > 0:
            # 연속된 미설치 층 확인
            consecutive = 1
            for i in range(1, len(floors_without_braces)):
                if floors_without_braces[i] == floors_without_braces[i-1] + 1:
                    consecutive += 1
                    if consecutive >= KoreanScaffoldRegulations.MAX_BRACE_VERTICAL_SPAN:
                        violations.append(f"가새 미설치: {consecutive}층 연속 (규정: {KoreanScaffoldRegulations.MAX_BRACE_VERTICAL_SPAN}단 이내)")
                        break
                else:
                    consecutive = 1

        return components, violations

    def _create_platforms_with_validation(self, num_bays, bay_width, depth, cumulative_heights, config):
        """🆕 발판 생성 + 규정 검증"""
        components = []
        violations = []

        safety_status = config.get('safety_status', 'safe')
        missing_rates = {'safe': 0.0, 'minor_defect': 0.2, 'major_defect': 0.4}
        missing_rate = missing_rates[safety_status]

        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            for bay in range(num_bays):
                if random.random() < missing_rate:
                    # 발판 누락
                    center_x = (bay + 0.5) * bay_width
                    center_y = depth / 2

                    # 누락 마커 (작은 포인트들)
                    points = []
                    for i in range(10):
                        x = center_x + random.uniform(-0.1, 0.1)
                        y = center_y + random.uniform(-0.1, 0.1)
                        points.append([x, y, z])

                    points = np.array(points)
                    bbox = self.calculate_bbox(points)

                    component = ScaffoldComponent(
                        name=f"누락발판_{floor_idx}층_{bay}베이_{self.instance_counter}",
                        semantic_id=10,  # 누락부분
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={'defect_type': 'missing_platform', 'floor': floor_idx, 'bay': bay}
                    )
                    components.append(component)
                    self.instance_counter += 1

                    violations.append(f"{floor_idx}층 {bay}베이 발판 누락")
                    continue

                # 정상 발판
                platform_center = np.array([(bay + 0.5) * bay_width, depth / 2, z])
                platform_width = min(bay_width * 0.9, random.uniform(0.35, 0.45))  # 폭 다양화
                platform_length = depth * 0.9

                # 🆕 규정 검증: 발판 폭
                width_violations = KoreanScaffoldRegulations.check_platform_width(platform_width)
                violations.extend(width_violations)

                platform_points = self.generate_platform_points(platform_center, platform_width, platform_length)

                if len(platform_points) > 0:
                    bbox = self.calculate_bbox(platform_points)

                    floor_name = "지면" if floor_idx == 0 else f"{floor_idx}층"
                    component = ScaffoldComponent(
                        name=f"발판_{floor_name}_{bay}베이_{self.instance_counter}",
                        semantic_id=3,
                        instance_id=self.instance_counter,
                        points=platform_points,
                        bbox=bbox,
                        metadata={'width': platform_width, 'floor': floor_idx, 'bay': bay}
                    )
                    components.append(component)
                    self.instance_counter += 1

        return components, violations

    def _create_safety_handrails(self, num_bays, bay_width, depth, cumulative_heights, config):
        """🆕 안전난간 생성 + 규정 검증"""
        components = []
        violations = []

        safety_status = config.get('safety_status', 'safe')

        # 안전난간 누락 확률
        if safety_status == 'safe':
            missing_prob = 0.0
        elif safety_status == 'minor_defect':
            missing_prob = 0.2
        else:
            missing_prob = 0.4

        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            if floor_idx == 0:  # 지면층은 안전난간 불필요
                continue

            if random.random() < missing_prob:
                violations.append(f"{floor_idx}층 안전난간 누락")
                continue

            # 상부난간 (90~120cm)
            top_rail_height = random.uniform(
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MIN,
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MAX
            )

            # 중간난간 (상부난간 절반 높이)
            mid_rail_height = top_rail_height / 2

            # 발끝막이판 (10cm)
            toe_board_height = KoreanScaffoldRegulations.TOE_BOARD_MIN_HEIGHT

            # 앞쪽 가장자리 안전난간
            for bay in range(num_bays):
                # 상부난간
                start_pos = np.array([bay * bay_width, 0, z])
                end_pos = np.array([(bay + 1) * bay_width, 0, z])

                top_rail = self.generate_safety_handrail(start_pos, end_pos, top_rail_height, 'top')
                if top_rail:
                    components.append(top_rail)

                # 중간난간
                mid_rail = self.generate_safety_handrail(start_pos, end_pos, mid_rail_height, 'mid')
                if mid_rail:
                    components.append(mid_rail)

                # 발끝막이판
                toe_board = self.generate_safety_handrail(start_pos, end_pos, toe_board_height, 'toe')
                if toe_board:
                    components.append(toe_board)

        return components, violations

    def _create_access_structures_simple(self, num_bays, bay_width, depth, cumulative_heights):
        """접근 구조 간소화 버전"""
        components = []

        num_floors = len(cumulative_heights) - 1

        # 계단 또는 사다리 (간단히)
        for floor_idx in range(num_floors):
            if floor_idx % 2 == 0:  # 짝수 층에 계단
                z_bottom = cumulative_heights[floor_idx]
                z_top = cumulative_heights[floor_idx + 1]

                stair_x = bay_width
                stair_y = depth / 2

                # 계단 레일
                for offset in [-0.3, 0.3]:
                    start_pos = np.array([stair_x - 0.5, stair_y + offset, z_bottom])
                    end_pos = np.array([stair_x + 0.5, stair_y + offset, z_top])

                    points = self.generate_pipe_points(start_pos, end_pos, 0.04, 40)

                    if len(points) > 0:
                        bbox = self.calculate_bbox(points)

                        component = ScaffoldComponent(
                            name=f"계단_{floor_idx}층_{self.instance_counter}",
                            semantic_id=6,
                            instance_id=self.instance_counter,
                            points=points,
                            bbox=bbox
                        )
                        components.append(component)
                        self.instance_counter += 1

        return components

    def _apply_diverse_defects(self, components, safety_status, config):
        """🆕 다양한 결함 적용"""
        defect_info = {
            'defect_types': [],
            'damaged_components': [],
            'violations': []
        }

        num_defects = 1 if safety_status == 'minor_defect' else 2

        # 결함 유형 pool
        defect_types = ['bent', 'cracked', 'corroded', 'loose_connection']

        beam_components = [c for c in components if c.semantic_id == 1]  # 수평재

        if not beam_components:
            return components, defect_info

        for _ in range(min(num_defects, len(beam_components))):
            target = random.choice(beam_components)
            beam_components.remove(target)

            defect_type = random.choice(defect_types)

            if defect_type == 'bent':
                # 휘어진 부재
                damaged = self._create_bent_beam(target)
                if damaged:
                    components = [c for c in components if c != target]
                    components.append(damaged)
                    defect_info['defect_types'].append('bent')
                    defect_info['damaged_components'].append(damaged.name)
                    defect_info['violations'].append(f"{target.name} 휨 변형")

            elif defect_type == 'cracked':
                # 균열 (포인트 일부 제거)
                damaged = self._create_cracked_beam(target)
                if damaged:
                    components = [c for c in components if c != target]
                    components.append(damaged)
                    defect_info['defect_types'].append('cracked')
                    defect_info['damaged_components'].append(damaged.name)
                    defect_info['violations'].append(f"{target.name} 균열 발생")

            elif defect_type == 'corroded':
                # 부식 (포인트 밀도 감소)
                damaged = self._create_corroded_beam(target)
                if damaged:
                    components = [c for c in components if c != target]
                    components.append(damaged)
                    defect_info['defect_types'].append('corroded')
                    defect_info['damaged_components'].append(damaged.name)
                    defect_info['violations'].append(f"{target.name} 부식")

            else:  # loose_connection
                # 느슨한 연결부 (메타데이터에만 표시)
                target.metadata = target.metadata or {}
                target.metadata['defect'] = 'loose_connection'
                defect_info['defect_types'].append('loose_connection')
                defect_info['damaged_components'].append(target.name)
                defect_info['violations'].append(f"{target.name} 연결부 느슨함")

        return components, defect_info

    def _create_bent_beam(self, original):
        """휘어진 부재 생성"""
        if len(original.points) == 0:
            return None

        points = original.points
        start_pos = points[np.argmin(points.sum(axis=1))]
        end_pos = points[np.argmax(points.sum(axis=1))]

        mid_point = (start_pos + end_pos) / 2
        mid_point[2] -= random.uniform(0.1, 0.25)

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['horizontal']
        points1 = self.generate_pipe_points(start_pos, mid_point, diameter, 30)
        points2 = self.generate_pipe_points(mid_point, end_pos, diameter, 30)

        if len(points1) > 0 and len(points2) > 0:
            all_points = np.vstack([points1, points2])
            bbox = self.calculate_bbox(all_points)

            return ScaffoldComponent(
                name=f"손상수평재_휨_{self.instance_counter}",
                semantic_id=9,
                instance_id=self.instance_counter,
                points=all_points,
                bbox=bbox,
                metadata={'defect_type': 'bent'}
            )

        return None

    def _create_cracked_beam(self, original):
        """균열 부재 생성 (포인트 일부 제거)"""
        if len(original.points) == 0:
            return None

        # 중간 부분 포인트 30% 제거
        num_points = len(original.points)
        crack_start = int(num_points * 0.4)
        crack_end = int(num_points * 0.6)

        # 균열 구간 제거
        cracked_points = np.vstack([
            original.points[:crack_start],
            original.points[crack_end:]
        ])

        bbox = self.calculate_bbox(cracked_points)

        return ScaffoldComponent(
            name=f"손상수평재_균열_{self.instance_counter}",
            semantic_id=9,
            instance_id=self.instance_counter,
            points=cracked_points,
            bbox=bbox,
            metadata={'defect_type': 'cracked'}
        )

    def _create_corroded_beam(self, original):
        """부식 부재 생성 (포인트 밀도 감소)"""
        if len(original.points) == 0:
            return None

        # 랜덤하게 50% 포인트만 유지
        num_points = len(original.points)
        keep_indices = np.random.choice(num_points, int(num_points * 0.5), replace=False)

        corroded_points = original.points[keep_indices]
        bbox = self.calculate_bbox(corroded_points)

        return ScaffoldComponent(
            name=f"손상수평재_부식_{self.instance_counter}",
            semantic_id=9,
            instance_id=self.instance_counter,
            points=corroded_points,
            bbox=bbox,
            metadata={'defect_type': 'corroded'}
        )

    def generate_shapellm_annotations(self, scene_id, components, config):
        """Create ShapeLLM-style annotations in English.

        This method generates a list of annotation dictionaries describing various
        tasks (referring segmentation, missing detection, safety assessment,
        damage detection, regulation compliance, spatial relations and
        component-specific checks) for a single scene. All questions and
        answers are written in English. Korean regulation violation messages
        are translated using the `_translate_violation` helper.
        """
        annotations: List[Dict] = []

        # Extract config details
        violations: List[str] = config.get('regulation_violations', [])
        defect_info = config.get('defect_info', {})
        safety_status = config.get('safety_status', 'safe')

        # Partition components by semantic id
        platforms = [c for c in components if c.semantic_id == 3]
        verticals = [c for c in components if c.semantic_id == 0]
        horizontals = [c for c in components if c.semantic_id == 1]
        handrails = [c for c in components if c.semantic_id == 8]
        missing_comps = [c for c in components if c.semantic_id == 10]
        damaged_comps = [c for c in components if c.semantic_id == 9]

        # ================================================================
        # 1️⃣ Referring Segmentation
        # ================================================================

        # Ask for locations of a subset of platforms (max 15)
        selected_platforms = random.sample(platforms, min(15, len(platforms))) if platforms else []
        for idx, comp in enumerate(selected_platforms, 1):
            bbox_str = self._format_bbox(comp.bbox) if comp.bbox is not None else "N/A"
            comp_name_en = self.class_names_en[comp.semantic_id]
            annotations.append({
                'id': f"{scene_id}_referring_platform_{idx:03d}",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\nWhere is the {comp_name_en} located?'
                    },
                    {
                        'from': 'gpt',
                        'value': f'The {comp_name_en} is located at {bbox_str}.'
                    }
                ],
                'task_type': 'referring_segmentation',
                'target_instance_id': comp.instance_id,
                'target_semantic_id': comp.semantic_id
            })

        # Ask for locations of a subset of vertical posts (max 5)
        selected_verticals = random.sample(verticals, min(5, len(verticals))) if verticals else []
        for idx, comp in enumerate(selected_verticals, 1):
            bbox_str = self._format_bbox(comp.bbox) if comp.bbox is not None else "N/A"
            comp_name_en = self.class_names_en[comp.semantic_id]
            annotations.append({
                'id': f"{scene_id}_referring_vertical_{idx:03d}",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\nWhere is the {comp_name_en} located?'
                    },
                    {
                        'from': 'gpt',
                        'value': f'The {comp_name_en} is located at {bbox_str}.'
                    }
                ],
                'task_type': 'referring_segmentation',
                'target_instance_id': comp.instance_id,
                'target_semantic_id': comp.semantic_id
            })

        # Ask for locations of a subset of safety rails (max 5)
        selected_handrails = random.sample(handrails, min(5, len(handrails))) if handrails else []
        for idx, comp in enumerate(selected_handrails, 1):
            bbox_str = self._format_bbox(comp.bbox) if comp.bbox is not None else "N/A"
            comp_name_en = self.class_names_en[comp.semantic_id]
            annotations.append({
                'id': f"{scene_id}_referring_handrail_{idx:03d}",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\nWhere is the {comp_name_en} located?'
                    },
                    {
                        'from': 'gpt',
                        'value': f'The {comp_name_en} is located at {bbox_str}.'
                    }
                ],
                'task_type': 'referring_segmentation',
                'target_instance_id': comp.instance_id,
                'target_semantic_id': comp.semantic_id
            })

        # ================================================================
        # 2️⃣ Missing Detection
        # ================================================================

        if missing_comps:
            # 2-1. Overall summary of missing components
            missing_info: List[str] = []
            for comp in missing_comps:
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                bbox_str = self._format_bbox(comp.bbox)
                missing_info.append(f"- Floor {floor}, Bay {bay}: {bbox_str}")

            annotations.append({
                'id': f"{scene_id}_missing_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any missing components? If so, provide their locations.'
                    },
                    {
                        'from': 'gpt',
                        'value': f"Yes, {len(missing_comps)} platforms are missing:\n" + '\n'.join(missing_info)
                    }
                ],
                'task_type': 'missing_detection_summary',
                'defect_type': 'missing_platform',
                'num_defects': len(missing_comps)
            })

            # 2-2. Missing components by floor
            floors_with_missing: Dict[str, List[ScaffoldComponent]] = {}
            for comp in missing_comps:
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                floors_with_missing.setdefault(floor, []).append(comp)

            for floor, comps_in_floor in floors_with_missing.items():
                floor_missing_info: List[str] = []
                for comp in comps_in_floor:
                    metadata = comp.metadata or {}
                    bay = metadata.get('bay', '?')
                    bbox_str = self._format_bbox(comp.bbox)
                    floor_missing_info.append(f"- Bay {bay}: {bbox_str}")

                annotations.append({
                    'id': f"{scene_id}_missing_floor_{floor}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nAre there any missing components on floor {floor}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"Yes, {len(comps_in_floor)} platforms are missing on floor {floor}:\n" + '\n'.join(floor_missing_info)
                        }
                    ],
                    'task_type': 'missing_detection_floor',
                    'target_floor': floor,
                    'num_defects': len(comps_in_floor)
                })

            # 2-3. Missing components by bay
            bays_with_missing: Dict[str, List[ScaffoldComponent]] = {}
            for comp in missing_comps:
                metadata = comp.metadata or {}
                bay = metadata.get('bay', '?')
                bays_with_missing.setdefault(bay, []).append(comp)

            for bay, comps_in_bay in bays_with_missing.items():
                bay_missing_info: List[str] = []
                for comp in comps_in_bay:
                    metadata = comp.metadata or {}
                    floor = metadata.get('floor', '?')
                    bbox_str = self._format_bbox(comp.bbox)
                    bay_missing_info.append(f"- Floor {floor}: {bbox_str}")

                annotations.append({
                    'id': f"{scene_id}_missing_bay_{bay}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nAre there any missing components in bay {bay}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"Yes, {len(comps_in_bay)} platforms are missing in bay {bay}:\n" + '\n'.join(bay_missing_info)
                        }
                    ],
                    'task_type': 'missing_detection_bay',
                    'target_bay': bay,
                    'num_defects': len(comps_in_bay)
                })

            # 2-4. Specific missing component questions (max 5)
            selected_missing = random.sample(missing_comps, min(5, len(missing_comps)))
            for idx, comp in enumerate(selected_missing, 1):
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                bbox_str = self._format_bbox(comp.bbox)

                annotations.append({
                    'id': f"{scene_id}_missing_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs there a platform at floor {floor}, bay {bay}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"No, the platform at floor {floor}, bay {bay} is missing. Expected location: {bbox_str}."
                        }
                    ],
                    'task_type': 'missing_detection_specific',
                    'target_floor': floor,
                    'target_bay': bay,
                    'target_instance_id': comp.instance_id
                })

        else:
            # Negative sample when no missing components exist
            annotations.append({
                'id': f"{scene_id}_missing_none",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any missing components?'
                    },
                    {
                        'from': 'gpt',
                        'value': 'No, all platforms are properly installed.'
                    }
                ],
                'task_type': 'missing_detection_negative',
                'num_defects': 0
            })

        # ================================================================
        # 3️⃣ Safety Assessment
        # ================================================================

        if violations:
            # Translate violations into English
            translated = [self._translate_violation(v) for v in violations]
            # Overall safety summary
            annotations.append({
                'id': f"{scene_id}_safety_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nProvide a general assessment of the structural safety of this scaffold.'
                    },
                    {
                        'from': 'gpt',
                        'value': f"This scaffold has {len(violations)} safety issues:\n" + '\n'.join([f'- {tv}' for tv in translated[:5]])
                    }
                ],
                'task_type': 'safety_assessment_summary',
                'safety_status': safety_status,
                'num_violations': len(violations)
            })

            # Categorize violations into common categories
            categories: Dict[str, List[str]] = {'spacing': [], 'brace': [], 'rail': [], 'platform': []}
            for v_en in translated:
                low = v_en.lower()
                if 'spacing' in low:
                    categories['spacing'].append(v_en)
                if 'brace' in low:
                    categories['brace'].append(v_en)
                if 'rail' in low or 'safety rail' in low:
                    categories['rail'].append(v_en)
                if 'platform' in low:
                    categories['platform'].append(v_en)

            # Generate specific safety questions per category
            for cat, vlist in categories.items():
                if not vlist:
                    continue
                # Human-friendly category name
                cat_readable = {
                    'spacing': 'spacing',
                    'brace': 'braces',
                    'rail': 'safety rail',
                    'platform': 'platform'
                }[cat]
                annotations.append({
                    'id': f"{scene_id}_safety_{cat}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nDoes the scaffold meet the safety requirements for {cat_readable}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"No, there are {len(vlist)} {cat_readable}-related issues:\n" + '\n'.join([f'- {tv}' for tv in vlist[:3]])
                        }
                    ],
                    'task_type': 'safety_assessment_specific',
                    'violation_category': cat,
                    'num_violations': len(vlist)
                })
        else:
            annotations.append({
                'id': f"{scene_id}_safety_pass",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nProvide a general assessment of the structural safety of this scaffold.'
                    },
                    {
                        'from': 'gpt',
                        'value': 'This scaffold is structurally safe. It meets all major safety criteria.'
                    }
                ],
                'task_type': 'safety_assessment_pass',
                'safety_status': 'safe',
                'num_violations': 0
            })

        # ================================================================
        # 4️⃣ Damage Detection
        # ================================================================

        if damaged_comps:
            # 4-1. Overall damage summary
            damage_info: List[str] = []
            for comp in damaged_comps:
                metadata = comp.metadata or {}
                defect_type = metadata.get('defect_type', 'unknown')
                defect_en = self.defect_en_map.get(defect_type, defect_type)
                bbox_str = self._format_bbox(comp.bbox)
                class_en = self.class_names_en[comp.semantic_id]
                damage_info.append(f"- {class_en}: {defect_en}, location: {bbox_str}")

            annotations.append({
                'id': f"{scene_id}_damage_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any damaged components?'
                    },
                    {
                        'from': 'gpt',
                        'value': f"Yes, {len(damaged_comps)} damaged components were found:\n" + '\n'.join(damage_info)
                    }
                ],
                'task_type': 'damage_detection_summary',
                'num_damaged': len(damaged_comps)
            })

            # 4-2. Damage questions by type
            damage_by_type: Dict[str, List[ScaffoldComponent]] = {}
            for comp in damaged_comps:
                metadata = comp.metadata or {}
                dtype = metadata.get('defect_type', 'unknown')
                damage_by_type.setdefault(dtype, []).append(comp)

            for dtype, comps_in_type in damage_by_type.items():
                dtype_en = self.defect_en_map.get(dtype, dtype)
                damage_list: List[str] = []
                for comp in comps_in_type:
                    bbox_str = self._format_bbox(comp.bbox)
                    class_en = self.class_names_en[comp.semantic_id]
                    damage_list.append(f"- {class_en}: {bbox_str}")
                annotations.append({
                    'id': f"{scene_id}_damage_{dtype}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nAre there any components with {dtype_en} damage?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"Yes, {len(comps_in_type)} components have {dtype_en} damage:\n" + '\n'.join(damage_list)
                        }
                    ],
                    'task_type': 'damage_detection_by_type',
                    'damage_type': dtype,
                    'num_damaged': len(comps_in_type)
                })

            # 4-3. Specific damaged component questions (max 3)
            selected_damaged = random.sample(damaged_comps, min(3, len(damaged_comps)))
            for idx, comp in enumerate(selected_damaged, 1):
                metadata = comp.metadata or {}
                dtype = metadata.get('defect_type', 'unknown')
                defect_en = self.defect_en_map.get(dtype, dtype)
                bbox_str = self._format_bbox(comp.bbox)
                class_en = self.class_names_en[comp.semantic_id]
                # Truncate the bbox string for readability
                short_bbox = bbox_str[:50] + '...' if len(bbox_str) > 50 else bbox_str
                annotations.append({
                    'id': f"{scene_id}_damage_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nPlease inspect this component. Location: {short_bbox}'
                        },
                        {
                            'from': 'gpt',
                            'value': f'This {class_en} has {defect_en} damage. It needs immediate replacement.'
                        }
                    ],
                    'task_type': 'damage_detection_specific',
                    'damage_type': dtype,
                    'target_instance_id': comp.instance_id
                })
        else:
            annotations.append({
                'id': f"{scene_id}_damage_none",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any damaged components?'
                    },
                    {
                        'from': 'gpt',
                        'value': 'No, all components are in good condition.'
                    }
                ],
                'task_type': 'damage_detection_negative',
                'num_damaged': 0
            })

        # ================================================================
        # 5️⃣ Regulation Compliance
        # ================================================================

        compliant = config.get('compliant', True)
        translated_violations = [self._translate_violation(v) for v in violations]

        # 5-1. Overall compliance question
        annotations.append({
            'id': f"{scene_id}_regulation_overall",
            'point': f"{scene_id}.npy",
            'conversations': [
                {
                    'from': 'human',
                    'value': '<point>\nDoes this scaffold comply with the Occupational Safety and Health Standards?'
                },
                {
                    'from': 'gpt',
                    'value': (
                        'Yes, it complies.\nAll safety criteria are met.'
                        if compliant
                        else 'No, it does not comply.\nViolations: ' + ', '.join(translated_violations[:3])
                    )
                }
            ],
            'task_type': 'regulation_compliance_overall',
            'compliant': compliant
        })

        # 5-2. Specific regulation questions by category
        regulation_questions_en = [
            ('spacing', 'Is the column spacing within the safety limits (1.85 m ledger, 1.50 m purlin)?'),
            ('platform', 'Is the platform width at least 40 cm?'),
            ('brace', 'Are braces installed within 5 spans?'),
            ('rail', 'Are safety rails (top, mid, toe boards) installed?'),
        ]

        for category, question in regulation_questions_en:
            # Determine which translated violations pertain to this category
            cat_violations = []
            for v in translated_violations:
                low = v.lower()
                if category == 'spacing' and 'spacing' in low:
                    cat_violations.append(v)
                elif category == 'platform' and 'platform' in low:
                    cat_violations.append(v)
                elif category == 'brace' and 'brace' in low:
                    cat_violations.append(v)
                elif category == 'rail' and ('rail' in low or 'safety rail' in low):
                    cat_violations.append(v)
            is_cat_compliant = len(cat_violations) == 0
            annotations.append({
                'id': f"{scene_id}_regulation_{category}",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<point>\n{question}'
                    },
                    {
                        'from': 'gpt',
                        'value': (
                            'Yes, it complies.'
                            if is_cat_compliant
                            else 'No, issues: ' + ', '.join(cat_violations[:2])
                        )
                    }
                ],
                'task_type': 'regulation_compliance_specific',
                'regulation_category': category,
                'compliant': is_cat_compliant
            })

        # ================================================================
        # 6️⃣ Spatial Relation
        # ================================================================

        # Heights of each floor
        if 'floor_heights' in config:
            floor_heights = config['floor_heights']
            if isinstance(floor_heights, (list, tuple)) and len(floor_heights) > 1:
                annotations.append({
                    'id': f"{scene_id}_spatial_floor_height",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': '<point>\nWhat are the heights of each floor?'
                        },
                        {
                            'from': 'gpt',
                            'value': 'Floor heights are: ' + ', '.join([f'Floor {i+1}: {h:.2f} m' for i, h in enumerate(floor_heights)])
                        }
                    ],
                    'task_type': 'spatial_relation_height',
                    'floor_heights': floor_heights
                })

        # Bay width question
        if 'bay_width' in config:
            bay_width = config['bay_width']
            annotations.append({
                'id': f"{scene_id}_spatial_bay_width",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nWhat is the width of the bay?'
                    },
                    {
                        'from': 'gpt',
                        'value': f'The bay width is {bay_width:.2f} m.'
                    }
                ],
                'task_type': 'spatial_relation_width',
                'bay_width': bay_width
            })

        # Overall structure size
        if 'num_bays' in config and 'num_floors' in config:
            num_bays = config['num_bays']
            num_floors = config['num_floors']
            annotations.append({
                'id': f"{scene_id}_spatial_structure_size",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nDescribe the overall size of this scaffold.'
                    },
                    {
                        'from': 'gpt',
                        'value': f'This scaffold has {num_bays} bays and {num_floors} floors.'
                    }
                ],
                'task_type': 'spatial_relation_structure',
                'num_bays': num_bays,
                'num_floors': num_floors
            })

        # ================================================================
        # 7️⃣ Component-Specific Checks
        # ================================================================

        if platforms:
            selected_for_check = random.sample(platforms, min(3, len(platforms)))
            for idx, comp in enumerate(selected_for_check, 1):
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                bay = metadata.get('bay', '?')
                # Determine if there are violations tied to this floor/bay from the original violation list
                related_viols = [self._translate_violation(v) for v in violations if f'{floor}층' in str(v) and f'{bay}베이' in str(v)]
                is_safe = len(related_viols) == 0
                annotations.append({
                    'id': f"{scene_id}_component_platform_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs the platform at floor {floor}, bay {bay} safe?'
                        },
                        {
                            'from': 'gpt',
                            'value': (
                                'Yes, it is safe.'
                                if is_safe
                                else 'Issues: ' + ', '.join(related_viols)
                            )
                        }
                    ],
                    'task_type': 'component_specific_check',
                    'target_instance_id': comp.instance_id,
                    'is_safe': is_safe
                })

        return annotations

    def _format_bbox(self, bbox):
        """Bbox를 문자열로 포맷"""
        # When no bounding box is available return a neutral English string
        if bbox is None:
            return "N/A"

        bbox_list = bbox.tolist()
        return str(bbox_list)

    def generate_scene_data(self, scene_id):
        """씬 데이터 생성 (개선 버전)"""
        # 인스턴스 카운터 초기화
        self.instance_counter = 1

        # 기본 config
        config = {
            'scene_id': scene_id
        }

        # 비계 구조 생성
        components, config = self.create_scaffold_with_regulations(config)

        if not components:
            return None

        # 포인트 병합
        all_points = []
        semantic_labels = []
        instance_labels = []

        for comp in components:
            if len(comp.points) == 0:
                continue

            all_points.append(comp.points)
            semantic_labels.extend([comp.semantic_id] * len(comp.points))
            instance_labels.extend([comp.instance_id] * len(comp.points))

        if not all_points:
            return None

        coord = np.vstack(all_points).astype(np.float32)
        semantic_gt = np.array(semantic_labels, dtype=np.int32)
        instance_gt = np.array(instance_labels, dtype=np.int32)

        # 포인트 수 조절 (50K-150K)
        target_points = random.randint(50000, 150000)
        current_points = len(coord)

        if current_points > target_points:
            indices = np.random.choice(current_points, target_points, replace=False)
            coord = coord[indices]
            semantic_gt = semantic_gt[indices]
            instance_gt = instance_gt[indices]
        elif current_points < target_points:
            needed = target_points - current_points
            indices = np.random.choice(current_points, needed, replace=True)

            extra_coord = coord[indices] + np.random.normal(0, 0.01, (needed, 3))
            extra_semantic = semantic_gt[indices]
            extra_instance = instance_gt[indices]

            coord = np.vstack([coord, extra_coord])
            semantic_gt = np.hstack([semantic_gt, extra_semantic])
            instance_gt = np.hstack([instance_gt, extra_instance])

        # 정규화 (v2 방식: center + scale + rotation)
        centroid = np.mean(coord, axis=0)
        coord_centered = coord - centroid

        # Scale: max distance from origin
        scale = float(np.linalg.norm(coord_centered, axis=1).max() + 1e-12)
        coord_scaled = coord_centered / scale

        # Optional Z-rotation (작은 각도로 augmentation)
        Rz_deg = float(np.random.uniform(-10.0, 10.0))
        theta = np.radians(Rz_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        coord_norm = (R @ coord_scaled.T).T

        # ✅ (N,3) -> (N,6): gray RGB 추가 (0~1 범위)
        gray_rgb = np.full((coord_norm.shape[0], 3), 0.5, dtype=np.float32)
        coord_norm = np.concatenate([coord_norm.astype(np.float32), gray_rgb], axis=1)

        # Normalization metadata 저장
        norm_params = {
            'centroid': centroid.tolist(),
            'scale': scale,
            'Rz_deg': Rz_deg
        }

        # Components에 bbox_norm 추가 (bbox_world는 이미 있음)
        for comp in components:
            if comp.bbox is not None:
                # bbox_world → bbox_norm 변환
                bbox_centered = (comp.bbox - centroid) / scale
                comp.bbox_norm = (R @ bbox_centered.T).T
            else:
                comp.bbox_norm = None

        # ShapeLLM annotations 생성
        annotations = self.generate_shapellm_annotations(scene_id, components, config)

        return {
            'coord': coord_norm,
            'semantic_gt': semantic_gt,
            'instance_gt': instance_gt,
            'scene_id': scene_id,
            'config': config,
            'annotations': annotations,
            'components': components,
            'norm_params': norm_params  # ← 추가
        }

    def save_for_shapellm(self, output_dir, num_scenes=1000, train_ratio=0.8, val_ratio=0.1):
        """ShapeLLM 형식으로 저장 (v2 features: meta, labels, split)"""
        output_path = Path(output_dir)
        pcs_dir = output_path / 'pcs'
        meta_dir = output_path / 'meta'
        labels_dir = output_path / 'labels'

        for d in [pcs_dir, meta_dir, labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        all_annotations = []
        all_scene_ids = []
        stats = defaultdict(int)

        print(f"🏗️ ShapeLLM용 비계 데이터 생성 시작 ({num_scenes} scenes)...")

        for i in range(num_scenes):
            scene_id = f"scaffold_{i:05d}"

            scene_data = self.generate_scene_data(scene_id)
            if scene_data is None:
                print(f"⚠️ Failed: {scene_id}")
                continue

            # .npy 파일 저장 (xyz + gray rgb => 6 channels)
            np.save(pcs_dir / f"{scene_id}.npy", scene_data['coord'].astype(np.float32))

            # Normalization metadata 저장 (v2 feature)
            scene_meta = {
                'scene_id': scene_id,
                'config': scene_data['config'],
                'norm_params': scene_data['norm_params']
            }
            with open(meta_dir / f"{scene_id}_meta.json", 'w', encoding='utf-8') as f:
                json.dump(scene_meta, f, indent=2, ensure_ascii=False)

            # Labels 저장 (bbox_world + bbox_norm) (v2 feature)
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

            # Annotations 수집
            all_annotations.extend(scene_data['annotations'])
            all_scene_ids.append(scene_id)

            # 통계
            stats['total'] += 1
            stats[scene_data['config']['safety_status']] += 1

            if (i + 1) % 100 == 0:
                print(f"  진행: {i + 1}/{num_scenes}")

        # Train/val/test split (v2 feature)
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

        # Annotations를 split별로 저장
        # Scene ID → annotations 매핑
        scene_to_annotations = defaultdict(list)
        for ann in all_annotations:
            # annotation의 scene_id는 'point' 필드에서 추출
            scene_id = ann['point'].replace('.npy', '')
            scene_to_annotations[scene_id].append(ann)

        for split_name in ['train', 'val', 'test']:
            split_annotations = []
            for scene_id in split[split_name]:
                split_annotations.extend(scene_to_annotations[scene_id])

            with open(output_path / f'instructions_{split_name}.json', 'w', encoding='utf-8') as f:
                json.dump(split_annotations, f, indent=2, ensure_ascii=False)

        # 메타데이터 저장
        metadata = {
            'num_scenes': stats['total'],
            'class_names': self.class_names,
            'safety_distribution': {
                'safe': stats['safe'],
                'minor_defect': stats['minor_defect'],
                'major_defect': stats['major_defect']
            },
            'total_annotations': len(all_annotations),
            'split': {
                'train': len(split['train']),
                'val': len(split['val']),
                'test': len(split['test'])
            }
        }

        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n" + "="*60)
        print("✅ ShapeLLM용 비계 데이터셋 생성 완료!")
        print("="*60)
        print(f"📁 Point Clouds: {pcs_dir} ({stats['total']}개)")
        print(f"📁 Meta: {meta_dir} ({stats['total']}개)")
        print(f"📁 Labels: {labels_dir} ({stats['total']}개)")
        print(f"📄 Annotations: instructions_{{train,val,test}}.json")
        print(f"📄 Split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
        print(f"\n📊 안전 상태 분포:")
        print(f"  ✅ 안전: {stats['safe']}개")
        print(f"  ⚠️ 경미: {stats['minor_defect']}개")
        print(f"  🚨 심각: {stats['major_defect']}개")

        return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🏗️ ShapeLLM용 비계 합성 데이터 생성 도구 (v2 enhanced)')
    parser.add_argument('--num_scenes', type=int, default=1000, help='생성할 scene 개수 (기본: 1000)')
    parser.add_argument('--output_dir', type=str, default='./playground/data/shapellm/scaffold_sft',
                        help='출력 디렉토리 경로 (기본: ./playground/data/shapellm/scaffold_sft)')
    parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드 (기본: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split 비율 (기본: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split 비율 (기본: 0.1)')
    args = parser.parse_args()

    generator = EnhancedScaffoldGenerator(random_seed=args.random_seed)
    stats = generator.save_for_shapellm(
        args.output_dir,
        num_scenes=args.num_scenes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    print("\n🎯 v2 주요 개선사항:")
    print("✅ 색상 정보 제거 (xyz 좌표만)")
    print("✅ 한국 산업안전보건기준 반영")
    print("✅ 다양한 결함 유형 (휨/균열/부식/느슨함)")
    print("✅ ShapeLLM annotation 형식")
    print("✅ Bbox 정보 포함 (world + normalized)")
    print("✅ 정규화 메타데이터 저장 (centroid, scale, Rz_deg)")
    print("✅ Train/val/test split 지원")
    print("✅ 5단계 학습 목표 지원")
    print("  1️⃣ Referring Segmentation")
    print("  2️⃣ 누락 감지")
    print("  3️⃣ 안정성 평가")
    print("  4️⃣ 손상 식별")
    print("  5️⃣ 규정 준수")
