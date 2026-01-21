"""
🏗️ 고도화된 비계 합성 데이터 생성 도구 (ShapeLLM용)
- 한국 산업안전보건기준 준수 (2025년 기준)
- 실제 시스템비계 규격 반영
- 점진적 학습 목표 지원 (Referring → 누락 감지 → 안정성 → 손상 → 규정)

This module defines a scaffold synthesis generator for ShapeLLM datasets.  All runtime
strings and identifiers used for file names or annotation content are written
in English to avoid issues when storing or loading JSON or other artefacts.  Only
docstrings and inline comments remain in Korean for documentation purposes.
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
    """비계 부품 정의 (Korean docstring)

    The core data structure representing a single scaffold component.  Runtime
    fields such as ``name`` are provided in English to ensure downstream
    consumers operate on ASCII-only data.  Comments and documentation are left
    untranslated.
    """
    name: str
    semantic_id: int
    instance_id: int
    points: np.ndarray  # [N, 3] coordinates (color removed)
    bbox: Optional[np.ndarray] = None  # [8, 3] bounding box corners (world coords)
    bbox_norm: Optional[np.ndarray] = None  # [8, 3] bounding box corners (normalized coords)
    metadata: Optional[Dict] = None  # 추가 메타데이터


class KoreanScaffoldRegulations:
    """한국 산업안전보건기준 (2025년)

    This class encodes the key numerical limits defined by Korean scaffold
    regulations.  The check methods return human‑readable English messages so
    that any violation stored in configuration or output files does not
    contain Korean text.
    """

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
    def check_column_spacing(cls, spacing_x: float, spacing_y: float) -> List[str]:
        """Validate column spacing and return violations in English.

        Args:
            spacing_x: spacing along the ledger direction (m).
            spacing_y: spacing along the purlin direction (m).

        Returns:
            A list of descriptive violation messages.  Messages are written in
            English for downstream processing.
        """
        violations: List[str] = []
        if spacing_x > cls.MAX_COLUMN_SPACING_LEDGER:
            violations.append(
                f"Column spacing exceeded in ledger direction: {spacing_x:.2f} m > {cls.MAX_COLUMN_SPACING_LEDGER} m"
            )
        if spacing_y > cls.MAX_COLUMN_SPACING_PURLIN:
            violations.append(
                f"Column spacing exceeded in purlin direction: {spacing_y:.2f} m > {cls.MAX_COLUMN_SPACING_PURLIN} m"
            )
        return violations

    @classmethod
    def check_platform_width(cls, width: float) -> List[str]:
        """Validate platform width and return violations in English.

        Args:
            width: platform width in metres.

        Returns:
            A list containing a single violation message if width is
            insufficient.  The message is in English.
        """
        if width < cls.MIN_PLATFORM_WIDTH:
            return [
                f"Platform width insufficient: {width:.2f} m < {cls.MIN_PLATFORM_WIDTH} m"
            ]
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
    """Generator for synthetic scaffold scenes with regulatory validation.

    This class encapsulates the generation of scaffold components, the
    application of random defects, the enforcement of safety regulations, and
    the creation of ShapeLLM‑style annotations.  All component names and
    messages used in runtime data (e.g., JSON) are produced in English.
    """

    def __init__(self, random_seed: Optional[int] = 42) -> None:
        """Initialize the scaffold generator.

        Args:
            random_seed: If provided, seeds NumPy and Python's ``random`` module
                for deterministic behaviour. If set to ``None``, the generator
                will not reseed the RNGs, allowing successive instances or calls
                to produce different results.
        """
        # Seed random number generators only if a seed is given. When
        # ``random_seed`` is ``None``, we leave the RNGs unchanged so that
        # multiple instances can produce varied scaffolds.
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Class names corresponding to semantic ids (English only)
        # 0: vertical post, 1: horizontal beam, 2: diagonal brace, 3: platform,
        # 4: base support, 5: connection, 6: stair, 7: ladder,
        # 8: safety rail, 9: damaged component, 10: missing part
        self.class_names: List[str] = [
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

        # A duplicate mapping for use in annotations; kept for clarity
        self.class_names_en: List[str] = list(self.class_names)

        # English defect names for damage detection
        self.defect_en_map: Dict[str, str] = {
            'bent': 'bent deformation',
            'cracked': 'crack',
            'corroded': 'corrosion',
            'loose_connection': 'loose connection'
        }

        self.instance_counter: int = 1
        self.components_registry: List[ScaffoldComponent] = []  # registry of all components

    def _translate_violation(self, violation: str) -> str:
        """Translate Korean regulation violation messages into English.

        This helper is retained for backward compatibility.  With the
        adjustments in ``KoreanScaffoldRegulations``, most violations are now
        generated in English.  The translator therefore primarily handles
        legacy strings that may still appear.

        Args:
            violation: A violation message in Korean or English.

        Returns:
            An English translation of the violation, or the original string
            if it does not match known patterns.
        """
        # Column spacing exceeded in ledger direction (띠장)
        if '띠장 방향 기둥 간격 초과' in violation:
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

    def calculate_bbox(self, points: np.ndarray) -> Optional[np.ndarray]:
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

    def generate_pipe_points(self, start_pos: np.ndarray, end_pos: np.ndarray, diameter: float, points_density: int = 100) -> np.ndarray:
        """파이프 형태의 포인트 생성 (색상 제거)"""
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return np.array([]).reshape(0, 3)

        direction = direction / length
        num_points = max(int(length * points_density), 10)
        num_points = min(num_points, 200)

        t_vals = np.linspace(0, 1, num_points)
        points: List[np.ndarray] = []

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

    def generate_platform_points(self, center: np.ndarray, width: float, length: float, points_density: int = 200) -> np.ndarray:
        """발판 포인트 생성 (색상 제거)"""
        points: List[List[float]] = []

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

    def generate_safety_handrail(self, start_pos: np.ndarray, end_pos: np.ndarray, height_offset: float, rail_type: str = 'top') -> Optional[ScaffoldComponent]:
        """🆕 안전난간 생성 (create safety rails).

        Creates a safety rail component between ``start_pos`` and ``end_pos`` with
        a given height offset above the scaffold floor.  The ``rail_type`` may
        be 'top', 'mid', or 'toe' to denote top rails, mid rails, or toe
        boards respectively.  Returns a ``ScaffoldComponent`` with English
        naming for use in downstream datasets.
        """
        rail_start = start_pos.copy()
        rail_end = end_pos.copy()
        rail_start[2] += height_offset
        rail_end[2] += height_offset

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['handrail']
        points = self.generate_pipe_points(rail_start, rail_end, diameter, 50)

        if len(points) > 0:
            bbox = self.calculate_bbox(points)

            component = ScaffoldComponent(
                name=f"safety_rail_{rail_type}_{self.instance_counter}",
                semantic_id=8,  # safety rail
                instance_id=self.instance_counter,
                points=points,
                bbox=bbox,
                metadata={'type': rail_type, 'height_offset': height_offset}
            )

            self.instance_counter += 1
            return component

        return None

    def create_scaffold_with_regulations(self, config: Dict) -> Tuple[List[ScaffoldComponent], Dict]:
        """🆕 규정 준수 검증을 포함한 비계 생성

        Generates a complete scaffold structure and checks it against the
        regulatory rules.  Returns the list of components and an updated
        configuration dictionary containing details of the scene.  All
        violation messages inserted into the configuration are in English.
        """
        components: List[ScaffoldComponent] = []
        violations: List[str] = []  # regulatory violations in English

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

        # 1. Base supports
        base_supports = self._create_base_supports_simple(num_bays, bay_width, scaffold_depth)
        components.extend(base_supports)

        # Determine safety status early so that missing rates can be computed.  If
        # ``safety_status`` is already present in the config (e.g., passed
        # externally) we reuse it; otherwise we sample a new status.  Once
        # determined, store it back into the config for downstream use.
        safety_status = config.get('safety_status', self._determine_safety_status())
        config['safety_status'] = safety_status

        # 2. Determine missing rate based on safety status.  This mapping
        # assigns the probability that any given structural component is
        # intentionally omitted when the scaffold is unsafe.  Only vertical
        # posts and horizontal beams use this value; platforms have their
        # own missing rates handled separately in _create_platforms_with_validation.
        missing_rates_map = {
            'safe': 0.0,
            'minor_defect': 0.2,
            'major_defect': 0.4,
        }
        # ``safety_status`` is now determined above and stored in config.  Use
        # it here to look up the missing rate for posts and beams.
        _miss_rate = missing_rates_map.get(safety_status, 0.0)

        # 2. Vertical posts
        posts = self._create_vertical_posts_simple(
            num_bays,
            bay_width,
            scaffold_depth,
            cumulative_heights,
            missing_rate=_miss_rate,
        )
        components.extend(posts)

        # 3. Horizontal beams
        beams = self._create_horizontal_beams_simple(
            num_bays,
            bay_width,
            scaffold_depth,
            cumulative_heights,
            missing_rate=_miss_rate,
        )
        components.extend(beams)

        # 4. Diagonal braces (with validation)
        diagonals, brace_violations = self._create_diagonal_braces_with_validation(
            num_bays, bay_width, scaffold_depth, cumulative_heights, num_floors
        )
        components.extend(diagonals)
        violations.extend(brace_violations)

        # 5. safety_status has already been determined and stored in config.

        # 6. Platforms (with validation).  Pass the updated config containing
        # safety_status so the missing_rate is computed correctly.
        platforms, platform_violations = self._create_platforms_with_validation(
            num_bays, bay_width, scaffold_depth, cumulative_heights, config
        )
        components.extend(platforms)
        violations.extend(platform_violations)

        # 7. Safety rails (based on regulations)
        handrails, handrail_violations = self._create_safety_handrails(
            num_bays, bay_width, scaffold_depth, cumulative_heights, config
        )
        components.extend(handrails)
        violations.extend(handrail_violations)

        # 8. Access structures (stairs/ladder simplified)
        access = self._create_access_structures_simple(num_bays, bay_width, scaffold_depth, cumulative_heights)
        components.extend(access)

        # 9. Diverse defects.  Apply only if safety_status is not safe.  Since
        # safety_status is already stored in config, we reuse it here.
        if safety_status != 'safe':
            components, defect_info = self._apply_diverse_defects(components, safety_status, config)
            violations.extend(defect_info.get('violations', []))
            config['defect_info'] = defect_info

        # Save violations as English only
        config['regulation_violations'] = violations
        config['compliant'] = len(violations) == 0

        # Register components
        self.components_registry = components

        return components, config

    def generate_diverse_floor_heights(self, num_floors: int, base_height_range: Tuple[float, float] = (1.8, 2.2)) -> List[float]:
        """층별 높이 다양화"""
        if random.random() < 0.7:
            uniform_height = random.uniform(*base_height_range)
            return [uniform_height] * num_floors
        else:
            floor_heights: List[float] = []
            for floor in range(num_floors):
                if floor == 0:
                    height = random.uniform(2.0, 2.5)
                else:
                    height = random.uniform(1.5, 2.3)
                floor_heights.append(height)
            return floor_heights

    def get_cumulative_heights(self, floor_heights: List[float]) -> List[float]:
        """누적 높이 계산"""
        cumulative = [0.1]
        for height in floor_heights:
            cumulative.append(cumulative[-1] + height)
        return cumulative

    def _determine_safety_status(self) -> str:
        """안전 상태 결정"""
        prob = random.random()
        if prob < 0.6:
            return 'safe'
        elif prob < 0.85:
            return 'minor_defect'
        else:
            return 'major_defect'

    # Simplified component creation functions (colour removed)
    def _create_base_supports_simple(self, num_bays: int, bay_width: float, depth: float) -> List[ScaffoldComponent]:
        """Generate base support components."""
        components: List[ScaffoldComponent] = []
        base_spec = ScaffoldSpecs.BASE_SUPPORT

        positions: List[List[float]] = []
        for i in range(num_bays + 1):
            for j in [0, depth]:
                positions.append([i * bay_width, j, -0.05])

        for pos in positions:
            base_center = np.array(pos)
            base_points = self._generate_base_support_points(base_center, base_spec['base_size'], base_spec['height'])

            if len(base_points) > 0:
                bbox = self.calculate_bbox(base_points)

                component = ScaffoldComponent(
                    name=f"base_support_{self.instance_counter}",
                    semantic_id=4,
                    instance_id=self.instance_counter,
                    points=base_points,
                    bbox=bbox
                )
                components.append(component)
                self.instance_counter += 1

        return components

    def _generate_base_support_points(self, center: np.ndarray, base_size: Tuple[float, float], height: float) -> np.ndarray:
        """하부받침 포인트 생성"""
        points: List[List[float]] = []
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

    def _create_vertical_posts_simple(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        missing_rate: float = 0.0,
    ) -> List[ScaffoldComponent]:
        """
        Generate vertical post components.  A portion of the posts may be
        intentionally omitted to simulate missing components.  When a post
        is missing, a small cluster of points is placed at the intended
        location to mark the missing area.  The ``missing_rate`` controls
        the probability that any given post is missing.

        Args:
            num_bays: number of bays along the X direction.
            bay_width: spacing between bays.
            depth: scaffold depth along Y direction.
            cumulative_heights: cumulative Z coordinates of floors.
            missing_rate: probability that a post is missing.
        Returns:
            List of ScaffoldComponent objects representing vertical posts
            and missing markers.
        """
        components: List[ScaffoldComponent] = []
        total_height = cumulative_heights[-1]

        # Collect positions for posts.  Each bay has two sides (front/back).
        positions: List[Tuple[int, int, List[float]]] = []
        for i in range(num_bays + 1):
            for side_idx, j in enumerate([0, depth]):
                # Store (bay index, side index, position)
                positions.append((i, side_idx, [i * bay_width, j, 0.1]))

        for bay_idx, side_idx, pos in positions:
            start_pos = np.array(pos)
            end_pos = start_pos + np.array([0, 0, total_height - 0.1])

            # Randomly decide if this post is missing.
            if missing_rate > 0.0 and random.random() < missing_rate:
                # Create a small cluster of points near the start position as a
                # marker for the missing post.  These points do not form a
                # complete pipe but provide a hint of the expected location.
                marker_points = []
                for _ in range(10):
                    noise = np.random.normal(0, 0.05, 3)
                    marker_points.append(start_pos + noise)
                marker_points = np.array(marker_points)
                bbox = self.calculate_bbox(marker_points)
                component = ScaffoldComponent(
                    name=f"missing_vertical_{bay_idx}_{side_idx}_{self.instance_counter}",
                    semantic_id=10,  # missing component
                    instance_id=self.instance_counter,
                    points=marker_points,
                    bbox=bbox,
                    metadata={'defect_type': 'missing_vertical', 'bay': bay_idx, 'side': side_idx}
                )
                components.append(component)
                self.instance_counter += 1
                continue

            # Otherwise create a full vertical post.
            diameter = ScaffoldSpecs.PIPE_DIAMETERS['vertical']
            points = self.generate_pipe_points(start_pos, end_pos, diameter)
            if len(points) > 0:
                bbox = self.calculate_bbox(points)
                component = ScaffoldComponent(
                    name=f"vertical_post_{self.instance_counter}",
                    semantic_id=0,
                    instance_id=self.instance_counter,
                    points=points,
                    bbox=bbox,
                    metadata={'bay': bay_idx, 'side': side_idx}
                )
                components.append(component)
                self.instance_counter += 1

        return components

    def _create_horizontal_beams_simple(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        missing_rate: float = 0.0,
    ) -> List[ScaffoldComponent]:
        """
        Generate horizontal beam components.  Each floor has beams in both X
        and Y directions.  A portion of beams may be intentionally omitted
        to simulate missing components.  Missing beams are represented by
        small clusters of points at the approximate midpoint of where the
        beam should be.  The ``missing_rate`` controls the probability
        that any given beam is missing.

        Args:
            num_bays: number of bays along X.
            bay_width: spacing between bays.
            depth: scaffold depth along Y.
            cumulative_heights: list of cumulative floor heights.
            missing_rate: probability that a beam is missing.
        Returns:
            List of ScaffoldComponent objects representing horizontal beams
            and missing markers.
        """
        components: List[ScaffoldComponent] = []

        # Enumerate floors and z-levels excluding the top cumulative height
        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            # X-direction beams: connect bay i to bay i+1 at both front/back sides
            for bay in range(num_bays):
                for side_idx, j in enumerate([0, depth]):
                    start_pos = np.array([bay * bay_width, j, z])
                    end_pos = np.array([(bay + 1) * bay_width, j, z])
                    # Decide if this beam is missing
                    if missing_rate > 0.0 and random.random() < missing_rate:
                        # Place a small cluster of points near the midpoint
                        mid_pos = (start_pos + end_pos) / 2.0
                        marker_points = []
                        for _ in range(10):
                            noise = np.random.normal(0, 0.05, 3)
                            marker_points.append(mid_pos + noise)
                        marker_points = np.array(marker_points)
                        bbox = self.calculate_bbox(marker_points)
                        comp = ScaffoldComponent(
                            name=f"missing_horizontal_X_{bay}_{side_idx}_{floor_idx}_{self.instance_counter}",
                            semantic_id=10,
                            instance_id=self.instance_counter,
                            points=marker_points,
                            bbox=bbox,
                            metadata={
                                'defect_type': 'missing_horizontal',
                                'orientation': 'X',
                                'floor': floor_idx,
                                'bay': bay,
                                'side': side_idx,
                            },
                        )
                        components.append(comp)
                        self.instance_counter += 1
                        continue
                    # Otherwise create the normal beam
                    diameter = ScaffoldSpecs.PIPE_DIAMETERS['horizontal']
                    points = self.generate_pipe_points(start_pos, end_pos, diameter)
                    if len(points) > 0:
                        bbox = self.calculate_bbox(points)
                        comp = ScaffoldComponent(
                            name=f"horizontal_beam_X_{self.instance_counter}",
                            semantic_id=1,
                            instance_id=self.instance_counter,
                            points=points,
                            bbox=bbox,
                            metadata={
                                'orientation': 'X',
                                'floor': floor_idx,
                                'bay': bay,
                                'side': side_idx,
                            },
                        )
                        components.append(comp)
                        self.instance_counter += 1

            # Y-direction beams: connect front to back at each column
            for col in range(num_bays + 1):
                start_pos = np.array([col * bay_width, 0, z])
                end_pos = np.array([col * bay_width, depth, z])
                # For Y beams we don't have two sides; use side_idx=0 for metadata
                if missing_rate > 0.0 and random.random() < missing_rate:
                    mid_pos = (start_pos + end_pos) / 2.0
                    marker_points = []
                    for _ in range(10):
                        noise = np.random.normal(0, 0.05, 3)
                        marker_points.append(mid_pos + noise)
                    marker_points = np.array(marker_points)
                    bbox = self.calculate_bbox(marker_points)
                    comp = ScaffoldComponent(
                        name=f"missing_horizontal_Y_{col}_{floor_idx}_{self.instance_counter}",
                        semantic_id=10,
                        instance_id=self.instance_counter,
                        points=marker_points,
                        bbox=bbox,
                        metadata={
                            'defect_type': 'missing_horizontal',
                            'orientation': 'Y',
                            'floor': floor_idx,
                            'column': col,
                            'side': 0,
                        },
                    )
                    components.append(comp)
                    self.instance_counter += 1
                    continue
                # Create normal Y-direction beam
                diameter = ScaffoldSpecs.PIPE_DIAMETERS['horizontal']
                points = self.generate_pipe_points(start_pos, end_pos, diameter)
                if len(points) > 0:
                    bbox = self.calculate_bbox(points)
                    comp = ScaffoldComponent(
                        name=f"horizontal_beam_Y_{self.instance_counter}",
                        semantic_id=1,
                        instance_id=self.instance_counter,
                        points=points,
                        bbox=bbox,
                        metadata={
                            'orientation': 'Y',
                            'floor': floor_idx,
                            'column': col,
                            'side': 0,
                        },
                    )
                    components.append(comp)
                    self.instance_counter += 1

        return components

    def _create_diagonal_braces_with_validation(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        num_floors: int
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """🆕 대각재 생성 + 규정 검증"""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        diameter = ScaffoldSpecs.PIPE_DIAMETERS['diagonal']

        floors_without_braces: List[int] = []

        for floor_idx in range(len(cumulative_heights) - 1):
            z_bottom = cumulative_heights[floor_idx]
            z_top = cumulative_heights[floor_idx + 1]

            # 60% chance of installing braces; simulate omissions for validation
            if random.random() < 0.6:
                # Install braces on front and back edges. In normal scaffolds, the diagonal braces are
                # typically oriented consistently (e.g., bottom-left to top-right), forming a Z-shape
                # pattern rather than crossing to make an X. We therefore generate braces that
                # connect adjacent vertical posts with the same orientation across all bays.
                for j in [0, depth]:
                    # Iterate through bays; connect each pair of adjacent posts to form a brace.
                    # We use step=2 to reduce the number of braces (every other bay) as in the
                    # original implementation. Adjust step to 1 if braces should appear on every bay.
                    for bay in range(0, num_bays, 2):
                        # Start at the bottom of the current bay post
                        start_pos = np.array([
                            bay * bay_width,
                            j,
                            z_bottom
                        ])
                        # End at the top of the adjacent (bay+1) post; this creates a consistent
                        # orientation (bottom-left → top-right) on both front (j=0) and back (j=depth) faces.
                        end_pos = np.array([
                            (bay + 1) * bay_width,
                            j,
                            z_top
                        ])

                        points = self.generate_pipe_points(start_pos, end_pos, diameter)

                        if len(points) > 0:
                            bbox = self.calculate_bbox(points)

                            component = ScaffoldComponent(
                                name=f"diagonal_brace_floor_{floor_idx}_{self.instance_counter}",
                                semantic_id=2,
                                instance_id=self.instance_counter,
                                points=points,
                                bbox=bbox,
                                metadata={'floor': floor_idx}
                            )
                            components.append(component)
                            self.instance_counter += 1
            else:
                floors_without_braces.append(floor_idx + 1)  # count floors starting from 1

        # 🆕 규정 검증: brace installation within specified vertical span
        if floors_without_braces:
            consecutive = 1
            for i in range(1, len(floors_without_braces)):
                if floors_without_braces[i] == floors_without_braces[i-1] + 1:
                    consecutive += 1
                    if consecutive >= KoreanScaffoldRegulations.MAX_BRACE_VERTICAL_SPAN:
                        violations.append(
                            f"Braces not installed: {consecutive} consecutive floors (allowed within {KoreanScaffoldRegulations.MAX_BRACE_VERTICAL_SPAN} spans)"
                        )
                        break
                else:
                    consecutive = 1

        return components, violations

    def _create_platforms_with_validation(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """🆕 발판 생성 + 규정 검증"""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        safety_status = config.get('safety_status', 'safe')
        missing_rates = {'safe': 0.0, 'minor_defect': 0.2, 'major_defect': 0.4}
        missing_rate = missing_rates[safety_status]

        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            for bay in range(num_bays):
                if random.random() < missing_rate:
                    # Missing platform
                    center_x = (bay + 0.5) * bay_width
                    center_y = depth / 2

                    # missing marker (small points)
                    points: List[List[float]] = []
                    for _ in range(10):
                        x = center_x + random.uniform(-0.1, 0.1)
                        y = center_y + random.uniform(-0.1, 0.1)
                        points.append([x, y, z])

                    points_arr = np.array(points)
                    bbox = self.calculate_bbox(points_arr)

                    component = ScaffoldComponent(
                        name=f"missing_platform_{floor_idx}_{bay}_{self.instance_counter}",
                        semantic_id=10,  # missing part
                        instance_id=self.instance_counter,
                        points=points_arr,
                        bbox=bbox,
                        metadata={'defect_type': 'missing_platform', 'floor': floor_idx, 'bay': bay}
                    )
                    components.append(component)
                    self.instance_counter += 1

                    violations.append(f"Missing platform at floor {floor_idx}, bay {bay}")
                    continue

                # Normal platform
                # In a safe scaffold, the working platform should cover the entire bay area (no gaps).
                # To achieve this, we generate a platform whose width spans the full bay width and
                # whose length spans the full scaffold depth. A small margin (5%) is removed to
                # avoid overlaps with rails or other elements.
                platform_center = np.array([
                    (bay + 0.5) * bay_width,
                    depth / 2,
                    z
                ])
                platform_width = bay_width * 0.95  # cover nearly the entire bay width
                platform_length = depth * 0.95     # cover nearly the entire scaffold depth

                # Ensure the platform meets the minimum width requirement defined in the regulations
                width_violations = KoreanScaffoldRegulations.check_platform_width(platform_width)
                violations.extend(width_violations)

                platform_points = self.generate_platform_points(platform_center, platform_width, platform_length)

                if len(platform_points) > 0:
                    bbox = self.calculate_bbox(platform_points)

                    floor_name = "ground" if floor_idx == 0 else f"floor_{floor_idx}"
                    component = ScaffoldComponent(
                        name=f"platform_{floor_name}_{bay}_{self.instance_counter}",
                        semantic_id=3,
                        instance_id=self.instance_counter,
                        points=platform_points,
                        bbox=bbox,
                        metadata={'width': platform_width, 'floor': floor_idx, 'bay': bay}
                    )
                    components.append(component)
                    self.instance_counter += 1

        return components, violations

    def _create_safety_handrails(
        self,
        num_bays: int,
        bay_width: float,
        depth: float,
        cumulative_heights: List[float],
        config: Dict
    ) -> Tuple[List[ScaffoldComponent], List[str]]:
        """🆕 안전난간 생성 + 규정 검증"""
        components: List[ScaffoldComponent] = []
        violations: List[str] = []

        safety_status = config.get('safety_status', 'safe')

        # Probability of missing safety rails
        if safety_status == 'safe':
            missing_prob = 0.0
        elif safety_status == 'minor_defect':
            missing_prob = 0.2
        else:
            missing_prob = 0.4

        for floor_idx, z in enumerate(cumulative_heights[:-1]):
            if floor_idx == 0:
                # ground floor does not require rails
                continue

            if random.random() < missing_prob:
                violations.append(f"Safety rail missing at floor {floor_idx}")
                continue

            # Top rail height between 0.90–1.20 m
            top_rail_height = random.uniform(
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MIN,
                KoreanScaffoldRegulations.TOP_RAIL_HEIGHT_MAX
            )

            # Mid rail height at half of the top rail
            mid_rail_height = top_rail_height / 2

            # Toe board height 10 cm
            toe_board_height = KoreanScaffoldRegulations.TOE_BOARD_MIN_HEIGHT

            # front edge rails
            for bay in range(num_bays):
                # Top rail
                start_pos = np.array([bay * bay_width, 0, z])
                end_pos = np.array([(bay + 1) * bay_width, 0, z])

                top_rail = self.generate_safety_handrail(start_pos, end_pos, top_rail_height, 'top')
                if top_rail:
                    components.append(top_rail)

                # Mid rail
                mid_rail = self.generate_safety_handrail(start_pos, end_pos, mid_rail_height, 'mid')
                if mid_rail:
                    components.append(mid_rail)

                # Toe board
                toe_board = self.generate_safety_handrail(start_pos, end_pos, toe_board_height, 'toe')
                if toe_board:
                    components.append(toe_board)

        return components, violations

    def _create_access_structures_simple(self, num_bays: int, bay_width: float, depth: float, cumulative_heights: List[float]) -> List[ScaffoldComponent]:
        """Simplified version of access structure generation (stairs only)."""
        components: List[ScaffoldComponent] = []

        num_floors = len(cumulative_heights) - 1

        for floor_idx in range(num_floors):
            if floor_idx % 2 == 0:
                z_bottom = cumulative_heights[floor_idx]
                z_top = cumulative_heights[floor_idx + 1]

                stair_x = bay_width
                stair_y = depth / 2

                # Two rails for the stair
                for offset in [-0.3, 0.3]:
                    start_pos = np.array([stair_x - 0.5, stair_y + offset, z_bottom])
                    end_pos = np.array([stair_x + 0.5, stair_y + offset, z_top])

                    points = self.generate_pipe_points(start_pos, end_pos, 0.04, 40)

                    if len(points) > 0:
                        bbox = self.calculate_bbox(points)

                        component = ScaffoldComponent(
                            name=f"stair_floor_{floor_idx}_{self.instance_counter}",
                            semantic_id=6,
                            instance_id=self.instance_counter,
                            points=points,
                            bbox=bbox
                        )
                        components.append(component)
                        self.instance_counter += 1

        return components

    def _apply_diverse_defects(self, components: List[ScaffoldComponent], safety_status: str, config: Dict) -> Tuple[List[ScaffoldComponent], Dict[str, List[str]]]:
        """🆕 다양한 결함 적용"""
        defect_info = {
            'defect_types': [],
            'damaged_components': [],
            'violations': []
        }

        num_defects = 1 if safety_status == 'minor_defect' else 2

        defect_types = ['bent', 'cracked', 'corroded', 'loose_connection']

        beam_components = [c for c in components if c.semantic_id == 1]  # horizontal beams

        if not beam_components:
            return components, defect_info

        for _ in range(min(num_defects, len(beam_components))):
            target = random.choice(beam_components)
            beam_components.remove(target)

            defect_type = random.choice(defect_types)

            if defect_type == 'bent':
                damaged = self._create_bent_beam(target)
                if damaged:
                    components = [c for c in components if c != target]
                    components.append(damaged)
                    defect_info['defect_types'].append('bent')
                    defect_info['damaged_components'].append(damaged.name)
                    defect_info['violations'].append(f"Bent deformation in {target.name}")

            elif defect_type == 'cracked':
                damaged = self._create_cracked_beam(target)
                if damaged:
                    components = [c for c in components if c != target]
                    components.append(damaged)
                    defect_info['defect_types'].append('cracked')
                    defect_info['damaged_components'].append(damaged.name)
                    defect_info['violations'].append(f"Crack in {target.name}")

            elif defect_type == 'corroded':
                damaged = self._create_corroded_beam(target)
                if damaged:
                    components = [c for c in components if c != target]
                    components.append(damaged)
                    defect_info['defect_types'].append('corroded')
                    defect_info['damaged_components'].append(damaged.name)
                    defect_info['violations'].append(f"Corrosion in {target.name}")

            else:  # loose_connection
                target.metadata = target.metadata or {}
                target.metadata['defect'] = 'loose_connection'
                defect_info['defect_types'].append('loose_connection')
                defect_info['damaged_components'].append(target.name)
                defect_info['violations'].append(f"Loose connection in {target.name}")

        return components, defect_info

    def _create_bent_beam(self, original: ScaffoldComponent) -> Optional[ScaffoldComponent]:
        """Create a bent horizontal beam."""
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
                name=f"damaged_horizontal_beam_bent_{self.instance_counter}",
                semantic_id=9,
                instance_id=self.instance_counter,
                points=all_points,
                bbox=bbox,
                metadata={'defect_type': 'bent'}
            )

        return None

    def _create_cracked_beam(self, original: ScaffoldComponent) -> Optional[ScaffoldComponent]:
        """Create a cracked horizontal beam by removing a section of points."""
        if len(original.points) == 0:
            return None

        num_points = len(original.points)
        crack_start = int(num_points * 0.4)
        crack_end = int(num_points * 0.6)

        cracked_points = np.vstack([
            original.points[:crack_start],
            original.points[crack_end:]
        ])

        bbox = self.calculate_bbox(cracked_points)

        return ScaffoldComponent(
            name=f"damaged_horizontal_beam_cracked_{self.instance_counter}",
            semantic_id=9,
            instance_id=self.instance_counter,
            points=cracked_points,
            bbox=bbox,
            metadata={'defect_type': 'cracked'}
        )

    def _create_corroded_beam(self, original: ScaffoldComponent) -> Optional[ScaffoldComponent]:
        """Create a corroded horizontal beam by reducing point density."""
        if len(original.points) == 0:
            return None

        num_points = len(original.points)
        keep_indices = np.random.choice(num_points, int(num_points * 0.5), replace=False)

        corroded_points = original.points[keep_indices]
        bbox = self.calculate_bbox(corroded_points)

        return ScaffoldComponent(
            name=f"damaged_horizontal_beam_corroded_{self.instance_counter}",
            semantic_id=9,
            instance_id=self.instance_counter,
            points=corroded_points,
            bbox=bbox,
            metadata={'defect_type': 'corroded'}
        )

    def generate_shapellm_annotations(self, scene_id: str, components: List[ScaffoldComponent], config: Dict) -> List[Dict]:
        """Create ShapeLLM-style annotations in English."""
        annotations: List[Dict] = []

        # Extract config details
        violations: List[str] = config.get('regulation_violations', [])
        defect_info: Dict = config.get('defect_info', {})
        safety_status: str = config.get('safety_status', 'safe')

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

            floors_with_missing: Dict[str, List[ScaffoldComponent]] = {}
            for comp in missing_comps:
                metadata = comp.metadata or {}
                floor = metadata.get('floor', '?')
                floors_with_missing.setdefault(str(floor), []).append(comp)

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

            bays_with_missing: Dict[str, List[ScaffoldComponent]] = {}
            for comp in missing_comps:
                metadata = comp.metadata or {}
                bay = metadata.get('bay', '?')
                bays_with_missing.setdefault(str(bay), []).append(comp)

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

        # 2-5. Specific positive component questions (max 5).
        # To train the model to respond "Yes" when a platform exists, we
        # generate a few positive examples asking about present platforms.  These
        # questions mirror the format of the missing-specific questions but with
        # an affirmative answer that includes the platform's bounding box.
        if platforms:
            import random as _rnd
            selected_present = _rnd.sample(platforms, min(5, len(platforms)))
            for idx, comp in enumerate(selected_present, 1):
                md = comp.metadata or {}
                floor_p = md.get('floor', '?')
                bay_p = md.get('bay', '?')
                bbox_str_p = self._format_bbox(comp.bbox) if comp.bbox is not None else "N/A"
                annotations.append({
                    'id': f"{scene_id}_missing_specific_positive_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs there a platform at floor {floor_p}, bay {bay_p}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"Yes, the platform at floor {floor_p}, bay {bay_p} is present. Location: {bbox_str_p}."
                        }
                    ],
                    'task_type': 'missing_detection_specific_positive',
                    'target_floor': floor_p,
                    'target_bay': bay_p,
                    'target_instance_id': comp.instance_id
                })

        # --------------------------
        # Missing detection for vertical posts
        # --------------------------
        # Collect missing and present vertical posts.  Missing vertical posts
        # have semantic_id == 10 and metadata['defect_type'] == 'missing_vertical'.
        # Present vertical posts have semantic_id == 0 and contain bay/side in
        # metadata for identification.
        missing_verticals = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_vertical']
        present_verticals = [c for c in verticals if (c.metadata or {}).get('bay') is not None]
        if missing_verticals:
            # Summary question listing all missing vertical posts.
            vertical_info: List[str] = []
            for comp in missing_verticals:
                md = comp.metadata or {}
                bay = md.get('bay', '?')
                side = md.get('side', '?')
                bbox_str = self._format_bbox(comp.bbox)
                vertical_info.append(f"- Bay {bay}, Side {side}: {bbox_str}")
            annotations.append({
                'id': f"{scene_id}_missing_vertical_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any missing vertical posts? If so, provide their locations.'
                    },
                    {
                        'from': 'gpt',
                        'value': f"Yes, {len(missing_verticals)} vertical posts are missing:\n" + '\n'.join(vertical_info)
                    }
                ],
                'task_type': 'missing_detection_vertical_summary',
                'num_defects': len(missing_verticals)
            })
            # Specific missing vertical post questions
            for idx, comp in enumerate(missing_verticals, 1):
                md = comp.metadata or {}
                bay = md.get('bay', '?')
                side = md.get('side', '?')
                bbox_str = self._format_bbox(comp.bbox)
                annotations.append({
                    'id': f"{scene_id}_missing_vertical_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs there a vertical post at bay {bay}, side {side}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"No, the vertical post at bay {bay}, side {side} is missing. Expected location: {bbox_str}."
                        }
                    ],
                    'task_type': 'missing_detection_vertical_specific',
                    'target_bay': bay,
                    'target_side': side,
                    'target_instance_id': comp.instance_id
                })
        else:
            # Negative question when no vertical posts are missing.
            annotations.append({
                'id': f"{scene_id}_missing_vertical_none",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any missing vertical posts?'
                    },
                    {
                        'from': 'gpt',
                        'value': 'No, all vertical posts are properly installed.'
                    }
                ],
                'task_type': 'missing_detection_vertical_none',
                'num_defects': 0
            })

        # Positive-specific vertical post questions
        if present_verticals:
            import random as _rnd
            selected_present_v = _rnd.sample(present_verticals, min(5, len(present_verticals)))
            for idx, comp in enumerate(selected_present_v, 1):
                md = comp.metadata or {}
                bay_p = md.get('bay', '?')
                side_p = md.get('side', '?')
                bbox_str_p = self._format_bbox(comp.bbox) if comp.bbox is not None else "N/A"
                annotations.append({
                    'id': f"{scene_id}_missing_vertical_specific_positive_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs there a vertical post at bay {bay_p}, side {side_p}?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"Yes, the vertical post at bay {bay_p}, side {side_p} is present. Location: {bbox_str_p}."
                        }
                    ],
                    'task_type': 'missing_detection_vertical_specific_positive',
                    'target_bay': bay_p,
                    'target_side': side_p,
                    'target_instance_id': comp.instance_id
                })

        # --------------------------
        # Missing detection for horizontal beams
        # --------------------------
        missing_horiz = [c for c in missing_comps if (c.metadata or {}).get('defect_type') == 'missing_horizontal']
        present_horiz = [c for c in horizontals if (c.metadata or {}).get('orientation') is not None]
        if missing_horiz:
            horiz_info: List[str] = []
            for comp in missing_horiz:
                md = comp.metadata or {}
                orientation = md.get('orientation', '?')
                floor_h = md.get('floor', '?')
                # Bay index for X-orientation or column index for Y-orientation
                bay_h = md.get('bay', md.get('column', '?'))
                side_h = md.get('side', '?')
                bbox_str = self._format_bbox(comp.bbox)
                label = 'Bay' if orientation == 'X' else 'Column'
                horiz_info.append(f"- Floor {floor_h}, {label} {bay_h}, Side {side_h}, {orientation}-dir: {bbox_str}")
            annotations.append({
                'id': f"{scene_id}_missing_horizontal_summary",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any missing horizontal beams? If so, provide their locations.'
                    },
                    {
                        'from': 'gpt',
                        'value': f"Yes, {len(missing_horiz)} horizontal beams are missing:\n" + '\n'.join(horiz_info)
                    }
                ],
                'task_type': 'missing_detection_horizontal_summary',
                'num_defects': len(missing_horiz)
            })
            # Specific missing horizontal beam questions
            for idx, comp in enumerate(missing_horiz, 1):
                md = comp.metadata or {}
                orientation = md.get('orientation', '?')
                floor_h = md.get('floor', '?')
                bay_h = md.get('bay', md.get('column', '?'))
                side_h = md.get('side', '?')
                bbox_str = self._format_bbox(comp.bbox)
                label_lower = 'bay' if orientation == 'X' else 'column'
                annotations.append({
                    'id': f"{scene_id}_missing_horizontal_specific_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs there a horizontal beam at floor {floor_h}, {label_lower} {bay_h}, side {side_h} in {orientation}-direction?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"No, the horizontal beam at floor {floor_h}, {label_lower} {bay_h}, side {side_h} in {orientation}-direction is missing. Expected location: {bbox_str}."
                        }
                    ],
                    'task_type': 'missing_detection_horizontal_specific',
                    'target_floor': floor_h,
                    'target_index': bay_h,
                    'target_side': side_h,
                    'target_orientation': orientation,
                    'target_instance_id': comp.instance_id
                })
        else:
            annotations.append({
                'id': f"{scene_id}_missing_horizontal_none",
                'point': f"{scene_id}.npy",
                'conversations': [
                    {
                        'from': 'human',
                        'value': '<point>\nAre there any missing horizontal beams?'
                    },
                    {
                        'from': 'gpt',
                        'value': 'No, all horizontal beams are properly installed.'
                    }
                ],
                'task_type': 'missing_detection_horizontal_none',
                'num_defects': 0
            })

        # Positive-specific horizontal beam questions
        if present_horiz:
            import random as _rnd
            selected_present_h = _rnd.sample(present_horiz, min(5, len(present_horiz)))
            for idx, comp in enumerate(selected_present_h, 1):
                md = comp.metadata or {}
                orientation = md.get('orientation', '?')
                floor_h = md.get('floor', '?')
                bay_h = md.get('bay', md.get('column', '?'))
                side_h = md.get('side', '?')
                bbox_str = self._format_bbox(comp.bbox) if comp.bbox is not None else "N/A"
                label_lower = 'bay' if orientation == 'X' else 'column'
                annotations.append({
                    'id': f"{scene_id}_missing_horizontal_specific_positive_{idx:03d}",
                    'point': f"{scene_id}.npy",
                    'conversations': [
                        {
                            'from': 'human',
                            'value': f'<point>\nIs there a horizontal beam at floor {floor_h}, {label_lower} {bay_h}, side {side_h} in {orientation}-direction?'
                        },
                        {
                            'from': 'gpt',
                            'value': f"Yes, the horizontal beam at floor {floor_h}, {label_lower} {bay_h}, side {side_h} in {orientation}-direction is present. Location: {bbox_str}."
                        }
                    ],
                    'task_type': 'missing_detection_horizontal_specific_positive',
                    'target_floor': floor_h,
                    'target_index': bay_h,
                    'target_side': side_h,
                    'target_orientation': orientation,
                    'target_instance_id': comp.instance_id
                })

        # ================================================================
        # 3️⃣ Safety Assessment
        # ================================================================

        if violations:
            translated = [self._translate_violation(v) for v in violations]
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

            for cat, vlist in categories.items():
                if not vlist:
                    continue
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
                            'value': (
                                'Yes, it complies.' if not vlist else 'No, there are ' + str(len(vlist)) + f' {cat_readable}-related issues:\n' + '\n'.join([f'- {tv}' for tv in vlist[:3]])
                            )
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

            selected_damaged = random.sample(damaged_comps, min(3, len(damaged_comps)))
            for idx, comp in enumerate(selected_damaged, 1):
                metadata = comp.metadata or {}
                dtype = metadata.get('defect_type', 'unknown')
                defect_en = self.defect_en_map.get(dtype, dtype)
                bbox_str = self._format_bbox(comp.bbox)
                class_en = self.class_names_en[comp.semantic_id]
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

        compliant: bool = config.get('compliant', True)
        translated_violations = [self._translate_violation(v) for v in violations]

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

        regulation_questions_en = [
            ('spacing', 'Is the column spacing within the safety limits (1.85 m ledger, 1.50 m purlin)?'),
            ('platform', 'Is the platform width at least 40 cm?'),
            ('brace', 'Are braces installed within 5 spans?'),
            ('rail', 'Are safety rails (top, mid, toe boards) installed?'),
        ]

        for category, question in regulation_questions_en:
            cat_violations: List[str] = []
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

        if 'bay_width' in config:
            bay_width_val = config['bay_width']
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
                        'value': f'The bay width is {bay_width_val:.2f} m.'
                    }
                ],
                'task_type': 'spatial_relation_width',
                'bay_width': bay_width_val
            })

        if 'num_bays' in config and 'num_floors' in config:
            n_bays = config['num_bays']
            n_floors = config['num_floors']
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
                        'value': f'This scaffold has {n_bays} bays and {n_floors} floors.'
                    }
                ],
                'task_type': 'spatial_relation_structure',
                'num_bays': n_bays,
                'num_floors': n_floors
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
                related_viols = [self._translate_violation(v) for v in violations if f'{floor}' in str(v) and f'{bay}' in str(v)]
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

    def _format_bbox(self, bbox: Optional[np.ndarray]) -> str:
        """Bbox를 문자열로 포맷"""
        if bbox is None:
            return "N/A"
        bbox_list = bbox.tolist()
        return str(bbox_list)

    def generate_scene_data(self, scene_id: str) -> Optional[Dict]:
        """Generate a single scene and its annotations."""
        self.instance_counter = 1

        config: Dict = {
            'scene_id': scene_id
        }

        components, config = self.create_scaffold_with_regulations(config)

        if not components:
            return None

        all_points: List[np.ndarray] = []
        semantic_labels: List[int] = []
        instance_labels: List[int] = []

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

        centroid = np.mean(coord, axis=0)
        coord_centered = coord - centroid
        scale = float(np.linalg.norm(coord_centered, axis=1).max() + 1e-12)
        coord_scaled = coord_centered / scale

        Rz_deg = float(np.random.uniform(-10.0, 10.0))
        theta = np.radians(Rz_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        coord_norm = (R @ coord_scaled.T).T

        gray_rgb = np.full((coord_norm.shape[0], 3), 0.5, dtype=np.float32)
        coord_norm = np.concatenate([coord_norm.astype(np.float32), gray_rgb], axis=1)

        norm_params = {
            'centroid': centroid.tolist(),
            'scale': scale,
            'Rz_deg': Rz_deg
        }

        for comp in components:
            if comp.bbox is not None:
                bbox_centered = (comp.bbox - centroid) / scale
                comp.bbox_norm = (R @ bbox_centered.T).T
            else:
                comp.bbox_norm = None

        annotations = self.generate_shapellm_annotations(scene_id, components, config)

        return {
            'coord': coord_norm,
            'semantic_gt': semantic_gt,
            'instance_gt': instance_gt,
            'scene_id': scene_id,
            'config': config,
            'annotations': annotations,
            'components': components,
            'norm_params': norm_params
        }

    def save_for_shapellm(
        self,
        output_dir: str,
        num_scenes: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, int]:
        """Save generated scenes in ShapeLLM format."""
        output_path = Path(output_dir)
        pcs_dir = output_path / 'pcs'
        meta_dir = output_path / 'meta'
        labels_dir = output_path / 'labels'

        for d in [pcs_dir, meta_dir, labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        all_annotations: List[Dict] = []
        all_scene_ids: List[str] = []
        stats: Dict[str, int] = defaultdict(int)

        print(f"🏗️ ShapeLLM dataset generation started ({num_scenes} scenes)...")

        for i in range(num_scenes):
            scene_id = f"scaffold_{i:05d}"
            scene_data = self.generate_scene_data(scene_id)
            if scene_data is None:
                print(f"⚠️ Failed: {scene_id}")
                continue

            np.save(pcs_dir / f"{scene_id}.npy", scene_data['coord'].astype(np.float32))

            scene_meta = {
                'scene_id': scene_id,
                'config': scene_data['config'],
                'norm_params': scene_data['norm_params']
            }
            with open(meta_dir / f"{scene_id}_meta.json", 'w', encoding='utf-8') as f:
                json.dump(scene_meta, f, indent=2, ensure_ascii=False)

            labels: List[Dict] = []
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

            all_annotations.extend(scene_data['annotations'])
            all_scene_ids.append(scene_id)

            stats['total'] += 1
            stats[scene_data['config']['safety_status']] += 1

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{num_scenes}")

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

        scene_to_annotations: Dict[str, List[Dict]] = defaultdict(list)
        for ann in all_annotations:
            scene_id_from_ann = ann['point'].replace('.npy', '')
            scene_to_annotations[scene_id_from_ann].append(ann)

        for split_name in ['train', 'val', 'test']:
            split_annotations: List[Dict] = []
            for scene_id in split[split_name]:
                split_annotations.extend(scene_to_annotations[scene_id])
            with open(output_path / f"instructions_{split_name}.json", 'w', encoding='utf-8') as f:
                json.dump(split_annotations, f, indent=2, ensure_ascii=False)

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
        print("✅ ShapeLLM dataset generation complete!")
        print("="*60)
        print(f"📁 Point Clouds: {pcs_dir} ({stats['total']} scenes)")
        print(f"📁 Meta: {meta_dir} ({stats['total']} scenes)")
        print(f"📁 Labels: {labels_dir} ({stats['total']} scenes)")
        print(f"📄 Annotations: instructions_{{train,val,test}}.json")
        print(f"📄 Split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
        print(f"\n📊 Safety distribution:")
        print(f"  ✅ safe: {stats['safe']} scenes")
        print(f"  ⚠️ minor: {stats['minor_defect']} scenes")
        print(f"  🚨 major: {stats['major_defect']} scenes")

        return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🏗️ ShapeLLM scaffold synthetic data generation tool (v2 enhanced)')
    parser.add_argument('--num_scenes', type=int, default=50, help='Number of scenes to generate (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='./playground/data/shapellm/scaffold_sft',
                        help='Output directory path (default: ./playground/data/shapellm/scaffold_sft)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio (default: 0.1)')
    args = parser.parse_args()

    generator = EnhancedScaffoldGenerator(random_seed=args.random_seed)
    stats = generator.save_for_shapellm(
        args.output_dir,
        num_scenes=args.num_scenes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    print("\n🎯 Key improvements in v2:")
    print("✅ Removed colour information (xyz only)")
    print("✅ Integrated Korean safety regulations")
    print("✅ Diverse defect types (bent/cracked/corroded/loose)")
    print("✅ ShapeLLM annotation format")
    print("✅ Included bounding box information (world + normalized)")
    print("✅ Saved normalization metadata (centroid, scale, Rz_deg)")
    print("✅ Supported train/val/test split")
    print("✅ Supported five learning objectives: referring, missing, safety, damage, regulation")