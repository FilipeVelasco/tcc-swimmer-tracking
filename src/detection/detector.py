"""
Camada de detecção.

- `Yolo11Detector`: wrapper fino em torno do modelo YOLO11 já treinado (TCC1).
  Usado quando `mode: "real"` na config.
- `SyntheticDetector`: usado quando `mode: "synthetic"`. Lê o ground-truth
  gerado por `scripts/generate_synthetic_data.py` e injeta ruído realista
  (jitter de bbox, falsos negativos, falsos positivos) para simular a saída
  de um detector real sem depender dos pesos treinados de verdade.

Ambos implementam a mesma interface (`BaseDetector.detect`), retornando um
`supervision.Detections`, então o resto do pipeline (tracker, pose, avaliação)
não sabe nem precisa saber qual dos dois está por trás.
"""
from __future__ import annotations

import csv
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv

from src.utils.config import DetectorConfig


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int) -> sv.Detections:
        """Roda a detecção em um frame e retorna as detecções no formato supervision."""
        raise NotImplementedError


class Yolo11Detector(BaseDetector):
    """Wrapper do modelo YOLO11 treinado no TCC1 para detecção de nadadores."""

    # Nomes de pesos oficiais do Ultralytics (baixam sozinhos se não existirem
    # localmente) — usados na fase de "bootstrap" com modelo genérico. Qualquer
    # outro caminho (ex.: 'models/meu_modelo.pt') é tratado como pesos
    # customizados/treinados e PRECISA existir localmente, para não mascarar
    # um caminho errado com um download silencioso de outra coisa.
    _OFFICIAL_WEIGHTS_PATTERN = re.compile(
        r"^yolo(v8|11)[nsmlx](-pose|-seg|-cls|-obb)?\.pt$"
    )

    def __init__(self, config: DetectorConfig):
        from ultralytics import YOLO  # import tardio: só é exigido no modo "real"

        weights = Path(config.weights_path)
        is_official = bool(self._OFFICIAL_WEIGHTS_PATTERN.match(weights.name))
        if not weights.exists() and not is_official:
            raise FileNotFoundError(
                f"Pesos do detector não encontrados em '{weights}'. "
                "Em modo 'real', aponte detector.weights_path no pipeline.yaml "
                "para o .pt treinado no TCC1 (ou use um nome oficial do "
                "Ultralytics, como 'yolo11n.pt', para a fase de bootstrap)."
            )
        self.model = YOLO(str(weights))  # baixa sozinho se for peso oficial e não existir
        self.config = config

    def detect(self, frame: np.ndarray, frame_idx: int) -> sv.Detections:
        results = self.model.predict(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            classes=self.config.classes,
            device=self.config.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(results)


class SyntheticDetector(BaseDetector):
    """
    Detector 'de mentira' para a fase hipotética: lê o GT sintético e simula
    o comportamento imperfeito de um detector real (jitter, FN, FP), permitindo
    testar tracking/pose/avaliação de ponta a ponta sem pesos treinados.
    """

    def __init__(
        self,
        gt_path: str,
        noise_std_px: float = 3.0,
        false_negative_rate: float = 0.04,
        false_positive_rate: float = 0.02,
        frame_size: tuple[int, int] = (1280, 720),
        seed: int = 7,
    ):
        self.rng = np.random.default_rng(seed)
        self.noise_std_px = noise_std_px
        self.fn_rate = false_negative_rate
        self.fp_rate = false_positive_rate
        self.frame_w, self.frame_h = frame_size
        self.boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        self._load_gt(gt_path)

    def _load_gt(self, gt_path: str) -> None:
        with open(gt_path, newline="") as f:
            for row in csv.reader(f):
                frame_idx = int(row[0]) - 1  # GT é 1-indexado
                x, y, w, h = (float(v) for v in row[2:6])
                self.boxes_by_frame[frame_idx].append((x, y, w, h))

    def detect(self, frame: np.ndarray, frame_idx: int) -> sv.Detections:
        xyxy, confidences, class_ids = [], [], []
        for (x, y, w, h) in self.boxes_by_frame.get(frame_idx, []):
            if self.rng.random() < self.fn_rate:
                continue  # falso negativo simulado
            jx, jy = self.rng.normal(0, self.noise_std_px, size=2)
            jw, jh = self.rng.normal(0, self.noise_std_px * 0.5, size=2)
            x1, y1 = x + jx, y + jy
            x2, y2 = x + w + jx + jw, y + h + jy + jh
            xyxy.append([x1, y1, x2, y2])
            confidences.append(float(np.clip(self.rng.normal(0.85, 0.08), 0.25, 0.99)))
            class_ids.append(0)

        if self.rng.random() < self.fp_rate:
            fx, fy = self.rng.uniform(0, self.frame_w - 80), self.rng.uniform(0, self.frame_h - 30)
            xyxy.append([fx, fy, fx + 75, fy + 30])
            confidences.append(float(self.rng.uniform(0.25, 0.5)))
            class_ids.append(0)

        if not xyxy:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )


def build_detector(mode: str, config: DetectorConfig, gt_path: str | None = None,
                    frame_size: tuple[int, int] = (1280, 720)) -> BaseDetector:
    """Factory: escolhe o detector real ou sintético conforme o modo do pipeline."""
    if mode == "real":
        return Yolo11Detector(config)
    if mode == "synthetic":
        if gt_path is None:
            raise ValueError("gt_path é obrigatório no modo 'synthetic'.")
        return SyntheticDetector(gt_path=gt_path, frame_size=frame_size)
    raise ValueError(f"Modo de pipeline desconhecido: {mode!r}")
