"""Carregamento tipado da configuração do pipeline (configs/pipeline.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    weights_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    classes: list[int] = field(default_factory=lambda: [0])
    min_box_width_px: float = 1.0
    min_box_height_px: float = 1.0
    expected_model_variant: str | None = None
    collect_timing: bool = True


@dataclass
class ReidFallbackConfig:
    enabled: bool = True
    max_lost_age_frames: int = 45
    similarity_threshold: float = 0.75
    histogram_bins: list[int] = field(default_factory=lambda: [8, 8, 8])


@dataclass
class TrackerConfig:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30
    lane_constraint_enabled: bool = True
    lane_switch_tolerance_frames: int = 3
    lane_reassociation_max_gap_frames: int = 90
    lane_reassociation_distance_factor: float = 2.5
    persistent_identity_enabled: bool = True
    identity_max_gap_frames: int = 90
    identity_distance_factor: float = 3.0
    identity_min_iou: float = 0.05
    identity_min_appearance_similarity: float = 0.55
    reid_fallback: ReidFallbackConfig = field(default_factory=ReidFallbackConfig)


@dataclass
class PoseConfig:
    weights_path: str = "yolo11n-pose.pt"
    conf_threshold: float = 0.3
    imgsz: int = 320
    device: str = "cpu"
    crop_margin: float = 0.15
    inference_interval_frames: int = 1
    stale_pose_max_frames: int = 6
    stale_pose_confidence_decay: float = 0.85


@dataclass
class StrokeCounterConfig:
    keypoint_indices: list[int] = field(default_factory=lambda: [9, 10])
    min_confidence: float = 0.3
    smoothing_window: int = 5
    min_peak_distance_sec: float = 0.4
    min_peak_prominence: float = 0.03
    max_gap_frames: int = 5


@dataclass
class EvaluationConfig:
    enabled: bool = True
    gt_path: str = "data/annotations/gt_mot.txt"
    metrics: list[str] = field(default_factory=list)
    iou_threshold_map: float = 0.5
    stroke_events_gt_path: str = "data/annotations/stroke_events_gt.csv"
    stroke_tolerance_frames: int = 5


@dataclass
class VideoConfig:
    source: str = "data/synthetic/synthetic_video.mp4"
    fps: int = 30
    width: int = 1280
    height: int = 720


@dataclass
class OutputConfig:
    dir: str = "outputs"
    save_annotated_video: bool = True
    save_metrics_json: bool = True
    save_stroke_counts_csv: bool = True
    save_mot_predictions: bool = True


@dataclass
class RoiConfig:
    enabled: bool = False
    # Pontos do polígono em coordenadas NORMALIZADAS (0.0-1.0), no sentido
    # horário ou anti-horário, cobrindo a área da piscina no frame.
    points: list[list[float]] = field(default_factory=list)
    lanes: list[list[list[float]]] = field(default_factory=list)


@dataclass
class LaneConfig:
    enabled: bool = False
    mode: str = "disabled"  # disabled | auto | manual
    number: int = 8
    min_water_ratio: float = 0.15
    min_line_length_ratio: float = 0.20
    max_line_gap: int = 25
    min_lane_width_ratio: float = 0.03
    allow_interpolation: bool = False
    calibration_frame: int = 0
    temporal_smoothing: float = 0.15


@dataclass
class PipelineConfig:
    mode: str
    video: VideoConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    pose: PoseConfig
    stroke_counter: StrokeCounterConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    roi: RoiConfig = field(default_factory=RoiConfig)
    lanes: LaneConfig = field(default_factory=LaneConfig)

    @staticmethod
    def load(path: str | Path) -> "PipelineConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return PipelineConfig(
            mode=raw["mode"],
            video=VideoConfig(**raw["video"]),
            detector=DetectorConfig(**raw["detector"]),
            tracker=TrackerConfig(
                **{k: v for k, v in raw["tracker"].items() if k != "reid_fallback"},
                reid_fallback=ReidFallbackConfig(**raw["tracker"]["reid_fallback"]),
            ),
            pose=PoseConfig(**raw["pose"]),
            stroke_counter=StrokeCounterConfig(**raw["stroke_counter"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            output=OutputConfig(**raw["output"]),
            roi=RoiConfig(**raw["roi"]) if "roi" in raw else RoiConfig(),
            lanes=LaneConfig(**raw["lanes"]) if "lanes" in raw else LaneConfig(),
        )

    def validate_inputs(self, *, require_video: bool = True, require_detector_weights: bool = True,
                        require_pose_weights: bool = True) -> None:
        """Validate the file and parameter contracts before starting inference."""
        errors: list[str] = []

        if self.mode not in {"synthetic", "real"}:
            errors.append(f"mode deve ser 'synthetic' ou 'real', recebido {self.mode!r}")
        if require_video and not Path(self.video.source).exists():
            errors.append(f"vídeo não encontrado: {self.video.source}")
        if self.video.fps <= 0 or self.video.width <= 0 or self.video.height <= 0:
            errors.append("video.fps, video.width e video.height devem ser positivos")
        if self.tracker.lane_switch_tolerance_frames < 0:
            errors.append("tracker.lane_switch_tolerance_frames não pode ser negativo")
        if self.tracker.lane_reassociation_max_gap_frames < 0:
            errors.append("tracker.lane_reassociation_max_gap_frames não pode ser negativo")
        if self.tracker.lane_reassociation_distance_factor <= 0:
            errors.append("tracker.lane_reassociation_distance_factor deve ser positivo")
        if self.tracker.identity_max_gap_frames < 0:
            errors.append("tracker.identity_max_gap_frames não pode ser negativo")
        if self.tracker.identity_distance_factor <= 0:
            errors.append("tracker.identity_distance_factor deve ser positivo")
        if not 0 <= self.tracker.identity_min_iou <= 1:
            errors.append("tracker.identity_min_iou deve estar no intervalo [0, 1]")
        if not 0 <= self.tracker.identity_min_appearance_similarity <= 1:
            errors.append("tracker.identity_min_appearance_similarity deve estar no intervalo [0, 1]")

        if not 0 < self.detector.conf_threshold <= 1:
            errors.append("detector.conf_threshold deve estar no intervalo (0, 1]")
        if not 0 < self.detector.iou_threshold <= 1:
            errors.append("detector.iou_threshold deve estar no intervalo (0, 1]")
        if not self.detector.classes or any(class_id < 0 for class_id in self.detector.classes):
            errors.append("detector.classes deve conter ao menos um índice não negativo")
        if self.detector.imgsz <= 0:
            errors.append("detector.imgsz deve ser positivo")
        if self.detector.min_box_width_px <= 0 or self.detector.min_box_height_px <= 0:
            errors.append("detector.min_box_width_px e min_box_height_px devem ser positivos")
        if require_detector_weights and self.mode == "real":
            weights = Path(self.detector.weights_path)
            official_name = weights.name.startswith(("yolo8", "yolo11")) and weights.suffix == ".pt"
            if not weights.exists() and not official_name:
                errors.append(f"pesos do detector não encontrados: {self.detector.weights_path}")

        if self.pose.conf_threshold <= 0 or self.pose.conf_threshold > 1:
            errors.append("pose.conf_threshold deve estar no intervalo (0, 1]")
        if self.pose.crop_margin < 0:
            errors.append("pose.crop_margin não pode ser negativo")
        if self.pose.inference_interval_frames <= 0:
            errors.append("pose.inference_interval_frames deve ser positivo")
        if self.pose.stale_pose_max_frames < 0:
            errors.append("pose.stale_pose_max_frames não pode ser negativo")
        if not 0 < self.pose.stale_pose_confidence_decay <= 1:
            errors.append("pose.stale_pose_confidence_decay deve estar no intervalo (0, 1]")
        if require_pose_weights and self.mode == "real" and not Path(self.pose.weights_path).exists():
            errors.append(f"pesos de pose não encontrados: {self.pose.weights_path}")

        if self.roi.enabled:
            if len(self.roi.points) < 3:
                errors.append("roi.points deve conter pelo menos três pontos")
            elif any(len(point) != 2 or not all(0 <= value <= 1 for value in point)
                     for point in self.roi.points):
                errors.append("roi.points deve usar pares normalizados entre 0 e 1")
        if self.lanes.mode not in {"disabled", "auto", "manual"}:
            errors.append("lanes.mode deve ser 'disabled', 'auto' ou 'manual'")
        if self.lanes.number < 1:
            errors.append("lanes.number deve ser positivo")
        if not 0 <= self.lanes.temporal_smoothing <= 1:
            errors.append("lanes.temporal_smoothing deve estar no intervalo [0, 1]")
        if not 0 <= self.lanes.min_water_ratio <= 1:
            errors.append("lanes.min_water_ratio deve estar no intervalo [0, 1]")
        if self.lanes.min_line_length_ratio <= 0:
            errors.append("lanes.min_line_length_ratio deve ser positivo")
        for lane_idx, lane in enumerate(self.roi.lanes, start=1):
            if len(lane) < 3 or any(len(point) != 2 or not all(0 <= value <= 1 for value in point)
                                     for point in lane):
                errors.append(f"roi.lanes[{lane_idx}] deve ser um polígono normalizado válido")

        if errors:
            raise ValueError("Configuração inválida:\n- " + "\n- ".join(errors))
