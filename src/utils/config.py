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
    reid_fallback: ReidFallbackConfig = field(default_factory=ReidFallbackConfig)


@dataclass
class PoseConfig:
    weights_path: str = "yolo11n-pose.pt"
    conf_threshold: float = 0.3
    imgsz: int = 320
    device: str = "cpu"
    crop_margin: float = 0.15


@dataclass
class StrokeCounterConfig:
    keypoint_indices: list[int] = field(default_factory=lambda: [9, 10])
    min_confidence: float = 0.3
    smoothing_window: int = 5
    min_peak_distance_sec: float = 0.4
    min_peak_prominence: float = 0.03


@dataclass
class EvaluationConfig:
    enabled: bool = True
    gt_path: str = "data/annotations/gt_mot.txt"
    metrics: list[str] = field(default_factory=list)
    iou_threshold_map: float = 0.5


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


@dataclass
class RoiConfig:
    enabled: bool = False
    # Pontos do polígono em coordenadas NORMALIZADAS (0.0-1.0), no sentido
    # horário ou anti-horário, cobrindo a área da piscina no frame.
    points: list[list[float]] = field(default_factory=list)


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
        )
