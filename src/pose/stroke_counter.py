"""
Contagem de braçadas (stroke counting) por track, a partir da oscilação
vertical dos punhos (keypoints COCO 9=esquerdo, 10=direito).

Abordagem: acumular a série temporal da posição Y do punho por track,
suavizar com média móvel e detectar picos (scipy.signal.find_peaks).
Cada pico = uma braçada daquele braço. É uma abordagem clássica e leve
(sem necessidade de modelo extra), adequada à fase hipotética e coerente
com a diretriz de manter tudo leve.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from src.utils.config import StrokeCounterConfig


@dataclass
class _WristSeries:
    frame_indices: list[int] = field(default_factory=list)
    y_values: list[float] = field(default_factory=list)


class StrokeCounter:
    def __init__(self, config: StrokeCounterConfig, fps: int):
        self.config = config
        self.fps = fps
        # track_id -> {keypoint_idx: _WristSeries}
        self._series: dict[int, dict[int, _WristSeries]] = defaultdict(
            lambda: {kp: _WristSeries() for kp in config.keypoint_indices}
        )

    def update(self, track_id: int, frame_idx: int,
               keypoints: dict[int, tuple[float, float, float]] | None) -> None:
        if keypoints is None:
            return
        for kp_idx in self.config.keypoint_indices:
            if kp_idx not in keypoints:
                continue
            x, y, conf = keypoints[kp_idx]
            if conf < self.config.min_confidence:
                continue
            series = self._series[track_id][kp_idx]
            series.frame_indices.append(frame_idx)
            series.y_values.append(y)

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        w = self.config.smoothing_window
        if w <= 1 or len(values) < w:
            return values
        kernel = np.ones(w) / w
        return np.convolve(values, kernel, mode="same")

    def count_for_track(self, track_id: int) -> dict[str, int | float]:
        """Retorna a contagem de braçadas por punho e o total combinado (média dos dois)."""
        result: dict[str, int | float] = {}
        per_wrist_counts = []
        for kp_idx in self.config.keypoint_indices:
            series = self._series[track_id][kp_idx]
            n_peaks = 0
            if len(series.y_values) >= 3:
                y = self._smooth(np.array(series.y_values, dtype=np.float64))
                bbox_scale = (y.max() - y.min()) or 1.0
                min_distance_frames = max(1, int(self.config.min_peak_distance_sec * self.fps))
                peaks, _ = find_peaks(
                    y,
                    distance=min_distance_frames,
                    prominence=self.config.min_peak_prominence * bbox_scale,
                )
                troughs, _ = find_peaks(
                    -y,
                    distance=min_distance_frames,
                    prominence=self.config.min_peak_prominence * bbox_scale,
                )
                n_peaks = len(peaks) + len(troughs)  # ciclo completo = 1 pico + 1 vale
                n_peaks = n_peaks // 2 if n_peaks >= 2 else n_peaks
            label = "left_wrist" if kp_idx == 9 else "right_wrist" if kp_idx == 10 else f"kp_{kp_idx}"
            result[label] = n_peaks
            per_wrist_counts.append(n_peaks)

        result["estimated_total_strokes"] = (
            round(float(np.mean(per_wrist_counts)), 1) if per_wrist_counts else 0.0
        )
        return result

    def summary(self) -> dict[int, dict[str, int | float]]:
        return {track_id: self.count_for_track(track_id) for track_id in self._series}
