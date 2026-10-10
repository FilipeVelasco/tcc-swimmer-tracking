"""Detecção automática e calibração geométrica das raias da piscina."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import LaneConfig


@dataclass
class LaneCalibration:
    """Resultado da calibração de raias em coordenadas normalizadas."""

    polygons: list[list[list[float]]]
    water_ratio: float
    line_count: int
    orientation: str
    confidence: float


class LaneDetector:
    """Estima regiões de raia usando água, bordas e linhas estruturais.

    A detecção automática é conservadora: quando a geometria não apresenta
    evidência suficiente, retorna zero polígonos em vez de criar raias falsas.
    """

    def __init__(self, config: LaneConfig):
        self.config = config
        self._previous: LaneCalibration | None = None

    def calibrate(self, frame: np.ndarray) -> LaneCalibration:
        """Calibra raias a partir de um frame de piscina."""
        if self.config.mode == "disabled" or not self.config.enabled:
            return LaneCalibration([], 0.0, 0, "none", 0.0)

        if self.config.mode == "manual":
            return LaneCalibration([], 0.0, 0, "manual", 1.0)

        water_mask = self._water_mask(frame)
        water_ratio = float(np.count_nonzero(water_mask)) / water_mask.size
        if water_ratio < self.config.min_water_ratio:
            return LaneCalibration([], water_ratio, 0, "none", 0.0)

        lines = self._detect_lines(water_mask)
        orientation, positions = self._select_lane_lines(lines, frame.shape[1], frame.shape[0])
        polygons = self._build_polygons(positions, orientation, frame.shape[1], frame.shape[0])
        confidence = min(1.0, water_ratio * 1.5) * min(1.0, len(positions) / 4)
        calibration = LaneCalibration(polygons, water_ratio, len(positions), orientation, confidence)

        if polygons and self._previous is not None and self.config.temporal_smoothing > 0:
            calibration = self._smooth(self._previous, calibration)
        if polygons:
            self._previous = calibration
        return calibration

    def _water_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([75, 25, 25], dtype=np.uint8)
        upper = np.array([135, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), dtype=np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _detect_lines(self, mask: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(mask, 50, 150)
        height, width = mask.shape[:2]
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(25, int(width * 0.08)),
            minLineLength=max(20, int(width * self.config.min_line_length_ratio)),
            maxLineGap=self.config.max_line_gap,
        )
        if lines is None:
            return np.empty((0, 4), dtype=np.int32)
        return np.asarray(lines, dtype=np.int32).reshape(-1, 4)

    def _select_lane_lines(
        self, lines: np.ndarray, width: int, height: int,
    ) -> tuple[str, list[float]]:
        candidates: dict[str, list[float]] = {"horizontal": [], "vertical": []}
        for x1, y1, x2, y2 in lines:
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < width * self.config.min_line_length_ratio:
                continue
            if abs(dx) >= abs(dy):
                candidates["horizontal"].append((y1 + y2) / (2 * height))
            else:
                candidates["vertical"].append((x1 + x2) / (2 * width))

        orientation = max(candidates, key=lambda key: len(candidates[key]))
        positions = sorted(candidates[orientation])
        clusters: list[float] = []
        for position in positions:
            if not clusters or abs(position - clusters[-1]) > self.config.min_lane_width_ratio / 2:
                clusters.append(position)
            else:
                clusters[-1] = (clusters[-1] + position) / 2
        if len(clusters) > self.config.number + 1:
            sample_indices = np.linspace(
                0, len(clusters) - 1, self.config.number + 1,
            ).round().astype(int)
            clusters = [clusters[index] for index in sample_indices]
        return orientation, clusters

    def _build_polygons(
        self, positions: list[float], orientation: str, width: int, height: int,
    ) -> list[list[list[float]]]:
        if len(positions) < 2:
            return []

        if len(positions) - 1 < self.config.number:
            if self.config.allow_interpolation:
                positions = np.linspace(
                    positions[0], positions[-1], self.config.number + 1,
                ).tolist()

        polygons = []
        for first, second in zip(positions[:-1], positions[1:]):
            if orientation == "horizontal":
                polygons.append([[0.0, first], [1.0, first], [1.0, second], [0.0, second]])
            else:
                polygons.append([[first, 0.0], [second, 0.0], [second, 1.0], [first, 1.0]])
        return polygons

    def _smooth(self, previous: LaneCalibration, current: LaneCalibration) -> LaneCalibration:
        if len(previous.polygons) != len(current.polygons):
            return current
        alpha = self.config.temporal_smoothing
        polygons = []
        for old_polygon, new_polygon in zip(previous.polygons, current.polygons):
            polygons.append([
                [round((1 - alpha) * old[0] + alpha * new[0], 6),
                 round((1 - alpha) * old[1] + alpha * new[1], 6)]
                for old, new in zip(old_polygon, new_polygon)
            ])
        return LaneCalibration(
            polygons, current.water_ratio, current.line_count,
            current.orientation, current.confidence,
        )

    @staticmethod
    def polygons_to_pixels(
        polygons: list[list[list[float]]], width: int, height: int,
    ) -> list[np.ndarray]:
        return [
            np.asarray([[x * width, y * height] for x, y in polygon], dtype=np.float32)
            for polygon in polygons
        ]