"""
Configuration module for scaffold data generation.

This module defines all configuration parameters for the scaffold data pipeline,
ensuring reproducibility and easy experimentation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import os


@dataclass
class ScaffoldGeometryConfig:
    """Scaffold geometry parameters based on Korean safety regulations."""

    # Bay configuration
    min_bays: int = 2
    max_bays: int = 4
    bay_width_range: tuple = (1.5, 2.0)  # meters

    # Depth configuration
    depth_range: tuple = (1.2, 1.8)  # meters

    # Floor configuration
    min_floors: int = 2
    max_floors: int = 4
    floor_height_range: tuple = (1.8, 2.2)  # meters

    # Pipe diameters (meters)
    vertical_diameter: float = 0.0486
    horizontal_diameter: float = 0.0427
    diagonal_diameter: float = 0.034

    # Point cloud parameters
    target_points_range: tuple = (50000, 150000)
    sample_points_num: int = 10000  # For training


@dataclass
class MissingConfig:
    """Missing component generation configuration."""

    # Global missing quota
    max_missing_per_scene: int = 4

    # Type-specific quotas
    max_missing_vertical: int = 1
    max_missing_horizontal: int = 2
    max_missing_platform: int = 1

    # Safety status distribution
    safe_ratio: float = 0.35       # No missing
    minor_defect_ratio: float = 0.35  # 1-2 missing
    major_defect_ratio: float = 0.30  # 3-4 missing

    # Missing rates by safety status
    minor_defect_rate: float = 0.1
    major_defect_rate: float = 0.2


@dataclass
class QuestionConfig:
    """Question generation configuration - CRITICAL for preventing data leakage."""

    # IMPORTANT: All questions use UNIFORM format to prevent data leakage
    # The model should NOT be able to guess Yes/No from question text alone
    #
    # V3 UPDATE: Removed {scaffold_spec} from questions!
    # Model must analyze point cloud to understand structure.

    # V3 mode - removes scaffold_spec from questions
    v3_no_text_shortcuts: bool = True

    # Summary question templates (V3: NO scaffold_spec, diverse templates)
    summary_question_templates: List[str] = field(default_factory=lambda: [
        "Analyze this scaffold structure. Are there any missing components? "
        "If so, provide their types and 3D locations.",

        "Inspect this scaffold for completeness. Report any missing parts "
        "with their positions and bounding boxes.",

        "Check this structure for gaps or missing elements. "
        "List all deficiencies found with their 3D locations.",

        "Examine the scaffold and identify any components that should be "
        "present but are missing. Provide 3D bounding boxes.",

        "Scan this scaffold structure and report any structural incompleteness "
        "with detailed location information."
    ])

    # Legacy template (for backward compatibility, not used in V3)
    summary_question_template: str = (
        "This is a {scaffold_spec}. "
        "Analyze the scaffold structure and determine if there are any missing components. "
        "If any components are missing, provide their types and 3D locations."
    )

    # Floor-specific question templates (V3: diverse)
    floor_question_templates: List[str] = field(default_factory=lambda: [
        "Check floor {floor_num} for any missing components. "
        "If found, provide their 3D bounding boxes.",

        "Inspect level {floor_num} of this scaffold. Are there any gaps? "
        "Report with 3D locations.",

        "Analyze floor {floor_num}. Report any missing parts with bounding boxes.",

        "Examine the {floor_num}th level. What components are absent? "
        "Provide 3D locations."
    ])

    floor_question_template: str = (
        "Examine floor {floor_num} of this scaffold. "
        "Are there any missing components on this floor? "
        "If so, identify them with their 3D bounding boxes."
    )

    # Bay-specific question templates (V3: diverse)
    bay_question_templates: List[str] = field(default_factory=lambda: [
        "Check bay {bay_num} for any missing components. "
        "If found, provide their 3D bounding boxes.",

        "Inspect bay section {bay_num}. Are there any gaps? "
        "Report with 3D locations.",

        "Analyze bay {bay_num}. Report any missing parts with bounding boxes.",

        "Examine bay number {bay_num}. What is missing? "
        "Provide 3D locations."
    ])

    bay_question_template: str = (
        "Inspect bay {bay_num} of this scaffold. "
        "Are there any missing components in this bay? "
        "If so, identify them with their 3D bounding boxes."
    )

    # Specific component question templates (V3: diverse)
    specific_question_templates: List[str] = field(default_factory=lambda: [
        "Is there a {component_type} at {location}? "
        "Provide the 3D bounding box if present.",

        "Check position {location}. Is the {component_type} present? "
        "Report with 3D location.",

        "Verify: Does a {component_type} exist at {location}? "
        "Provide bounding box.",

        "Inspect location {location} for {component_type} presence. "
        "Report with 3D coordinates."
    ])

    specific_question_template: str = (
        "Check if there is a {component_type} at {location}. "
        "Provide your answer with the component's 3D bounding box if present, "
        "or indicate if it is missing."
    )

    # Component type templates (V3: diverse)
    vertical_question_templates: List[str] = field(default_factory=lambda: [
        "Are all vertical posts present in this scaffold? "
        "Report any missing with 3D locations.",

        "Check for missing vertical supports. "
        "Provide bounding boxes for any gaps.",

        "Inspect the vertical structural elements. Any missing? "
        "List with 3D coordinates.",

        "Analyze vertical posts. Report any gaps with locations."
    ])

    vertical_question_template: str = (
        "Analyze the vertical posts of this scaffold. "
        "Are there any missing vertical posts? "
        "If so, provide their locations with 3D bounding boxes."
    )

    horizontal_question_templates: List[str] = field(default_factory=lambda: [
        "Are all horizontal beams present in this scaffold? "
        "Report any missing with 3D locations.",

        "Check for missing horizontal members. "
        "Provide bounding boxes for any gaps.",

        "Inspect the horizontal structural elements. Any missing? "
        "List with 3D coordinates.",

        "Analyze horizontal beams. Report any gaps with locations."
    ])

    horizontal_question_template: str = (
        "Analyze the horizontal beams of this scaffold. "
        "Are there any missing horizontal beams? "
        "If so, provide their locations with 3D bounding boxes."
    )


@dataclass
class AnswerConfig:
    """Answer template configuration for consistent formatting."""

    # Yes answer template (with missing components)
    yes_summary_template: str = (
        "Expected structure: {scaffold_spec} should have {expected_verticals} vertical posts, "
        "{expected_horizontals} horizontal beams, and {expected_platforms} platforms.\n"
        "Actual structure: {actual_verticals} vertical posts, {actual_horizontals} horizontal beams, "
        "{actual_platforms} platforms.\n"
        "Missing components detected ({num_missing} total):\n{missing_details}"
    )

    # No answer template (no missing components)
    no_summary_template: str = (
        "Expected structure: {scaffold_spec} should have {expected_verticals} vertical posts, "
        "{expected_horizontals} horizontal beams, and {expected_platforms} platforms.\n"
        "Actual structure: {actual_verticals} vertical posts, {actual_horizontals} horizontal beams, "
        "{actual_platforms} platforms.\n"
        "No missing components detected. All scaffold elements are properly installed."
    )

    # Component location format
    bbox_format: str = "[[{x1:.3f}, {y1:.3f}, {z1:.3f}], ..., [{x8:.3f}, {y8:.3f}, {z8:.3f}]]"


@dataclass
class Stage1Config:
    """Stage 1 (Feature Alignment) specific configuration."""

    # Caption templates for Stage 1 pretraining
    caption_templates: List[str] = field(default_factory=lambda: [
        "A {num_bays}-bay, {num_floors}-floor scaffold structure with {num_verticals} vertical posts "
        "and {num_horizontals} horizontal beams.",

        "This is a multi-level scaffold measuring approximately {width:.1f}m x {depth:.1f}m x {height:.1f}m, "
        "featuring {num_platforms} work platforms.",

        "A standard construction scaffold with {num_bays} bays and {num_floors} levels, "
        "constructed using standard steel pipes.",

        "Industrial scaffold structure with vertical posts at {bay_width:.1f}m intervals, "
        "supporting {num_floors} floor levels.",

        "A {safety_status} scaffold assembly with {total_components} structural components "
        "including posts, beams, and platforms."
    ])

    # Number of captions per scene
    captions_per_scene: int = 3


@dataclass
class DatasetConfig:
    """Dataset split and output configuration."""

    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Output paths (V3: default to scaffold_v3)
    output_dir: str = "./playground/data/shapellm/scaffold_v3"

    # File names
    stage1_train_file: str = "stage1_caption_train.json"
    stage1_val_file: str = "stage1_caption_val.json"
    stage2_train_file: str = "stage2_sft_train.json"
    stage2_val_file: str = "stage2_sft_val.json"
    test_questions_file: str = "test_questions.jsonl"
    test_gt_file: str = "test_gt.jsonl"

    # Point cloud directory
    pcs_dir: str = "pcs"


@dataclass
class AblationConfig:
    """Ablation study configuration for validating model behavior."""

    # Generate ablation test sets
    generate_ablation: bool = True

    # Ablation types
    ablation_types: List[str] = field(default_factory=lambda: [
        "random_noise",      # Replace point cloud with random noise
        "zeros",             # Replace with zero point cloud
        "shuffled",          # Shuffle point cloud points
        "different_scaffold" # Use different scaffold for same question
    ])

    # Number of samples per ablation type
    samples_per_type: int = 100


@dataclass
class ScaffoldConfig:
    """Master configuration combining all sub-configurations."""

    geometry: ScaffoldGeometryConfig = field(default_factory=ScaffoldGeometryConfig)
    missing: MissingConfig = field(default_factory=MissingConfig)
    question: QuestionConfig = field(default_factory=QuestionConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    # Global settings
    random_seed: int = 42
    num_scenes: int = 3000

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ScaffoldConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses

        def dataclass_to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(v) for v in obj]
            else:
                return obj

        with open(yaml_path, 'w') as f:
            yaml.dump(dataclass_to_dict(self), f, default_flow_style=False)

    def validate(self) -> List[str]:
        """Validate configuration and return any warnings."""
        warnings = []

        # Check split ratios
        total_ratio = self.dataset.train_ratio + self.dataset.val_ratio + self.dataset.test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            warnings.append(f"Split ratios sum to {total_ratio}, expected 1.0")

        # Check missing quotas
        if self.missing.max_missing_per_scene < (
            self.missing.max_missing_vertical +
            self.missing.max_missing_horizontal +
            self.missing.max_missing_platform
        ):
            warnings.append("Type-specific missing quotas exceed global quota")

        # Check safety status distribution
        status_total = (
            self.missing.safe_ratio +
            self.missing.minor_defect_ratio +
            self.missing.major_defect_ratio
        )
        if abs(status_total - 1.0) > 0.01:
            warnings.append(f"Safety status ratios sum to {status_total}, expected 1.0")

        return warnings
