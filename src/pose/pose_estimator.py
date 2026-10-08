"""
Estimação de pose por track.

Para cada bbox rastreada, recorta a região (com uma margem) e roda o YOLO11n-pose
nela — em vez de rodar em frame inteiro — porque:
  1) mantém o pipeline leve (imagem menor -> inferência mais rápida)
  2) evita ambiguidade de associar keypoints entre nadadores em raias próximas

Retorna os keypoints já traduzidos de volta para coordenadas do frame original.

Skeleton COCO (17 keypoints) usado pelo YOLO11-pose:
  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

Também expõe `SyntheticPoseEstimator`, que lê os keypoints de punho do GT
sintético (gerados junto com o vídeo) para a fase hipotética — assim o
stroke counter pode ser testado e validado antes de existir um modelo de
pose real treinado (a seleção do modelo de pose foi propositalmente
deixada para o TCC2, conforme documentado).
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from src.utils.config import PoseConfig

LEFT_WRIST, RIGHT_WRIST = 9, 10


class BasePoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray, bbox_xyxy: np.ndarray, frame_idx: int,
                 track_id: int) -> dict[int, tuple[float, float, float]] | None:
        """Retorna {keypoint_idx: (x, y, confidence)} em coordenadas do frame, ou None."""
        raise NotImplementedError


class Yolo11PoseEstimator(BasePoseEstimator):
    def __init__(self, config: PoseConfig):
        from ultralytics import YOLO  # import tardio

        self.model = YOLO(config.weights_path)
        self.config = config

    def estimate(self, frame: np.ndarray, bbox_xyxy: np.ndarray, frame_idx: int,
                 track_id: int) -> dict[int, tuple[float, float, float]] | None:
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        margin = self.config.crop_margin
        cx1 = max(0, int(x1 - w * margin))
        cy1 = max(0, int(y1 - h * margin))
        cx2 = min(w_frame, int(x2 + w * margin))
        cy2 = min(h_frame, int(y2 + h * margin))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        results = self.model.predict(
            crop, conf=self.config.conf_threshold, imgsz=self.config.imgsz,
            device=self.config.device, verbose=False,
        )[0]

        if results.keypoints is None or len(results.keypoints) == 0:
            return None

        # Se houver mais de uma pessoa na crop (raro, mas possível), pega a maior detecção
        # (mais próxima do centro/tamanho da bbox original do tracker).
        boxes = results.boxes
        best_idx = 0
        if boxes is not None and len(boxes) > 1:
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best_idx = int(areas.argmax())

        kp_xy = results.keypoints.xy[best_idx].cpu().numpy()
        kp_conf = results.keypoints.conf[best_idx].cpu().numpy() if results.keypoints.conf is not None \
            else np.ones(len(kp_xy))

        keypoints = {}
        for idx, ((x, y), c) in enumerate(zip(kp_xy, kp_conf)):
            keypoints[idx] = (float(x + cx1), float(y + cy1), float(c))
        return keypoints


class SyntheticPoseEstimator(BasePoseEstimator):
    """Lê os keypoints de punho do GT sintético (fase hipotética, sem modelo de pose real)."""

    def __init__(self, gt_keypoints_path: str):
        self.by_frame_track: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        with open(gt_keypoints_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row["frame"]) - 1
                track_id = int(row["track_id"])
                self.by_frame_track[(frame_idx, track_id)] = (
                    float(row["left_wrist_x"]), float(row["left_wrist_y"]),
                    float(row["right_wrist_x"]), float(row["right_wrist_y"]),
                )

    def estimate(self, frame: np.ndarray, bbox_xyxy: np.ndarray, frame_idx: int,
                 track_id: int) -> dict[int, tuple[float, float, float]] | None:
        entry = self.by_frame_track.get((frame_idx, track_id))
        if entry is None:
            return None
        lx, ly, rx, ry = entry
        return {LEFT_WRIST: (lx, ly, 0.95), RIGHT_WRIST: (rx, ry, 0.95)}


def build_pose_estimator(mode: str, config: PoseConfig, gt_keypoints_path: str | None = None) -> BasePoseEstimator:
    if mode == "real":
        return Yolo11PoseEstimator(config)
    if mode == "synthetic":
        if gt_keypoints_path is None:
            raise ValueError("gt_keypoints_path é obrigatório no modo 'synthetic'.")
        return SyntheticPoseEstimator(gt_keypoints_path)
    raise ValueError(f"Modo de pipeline desconhecido: {mode!r}")
