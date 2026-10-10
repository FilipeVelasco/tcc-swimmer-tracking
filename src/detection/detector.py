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
import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import supervision as sv

from src.utils.config import DetectorConfig


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int) -> sv.Detections:
        """Roda a detecção em um frame e retorna as detecções no formato supervision."""
        raise NotImplementedError


@dataclass
class DetectionBatch:
    """Resultado completo de uma inferência em um frame."""

    frame_id: int
    frame_width: int
    frame_height: int
    boxes_xyxy: np.ndarray
    boxes_cxcywh: np.ndarray
    boxes_normalized: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    detections: sv.Detections
    timings_ms: dict[str, float]

    @property
    def count(self) -> int:
        """Retorna a quantidade de detecções válidas."""
        return int(len(self.boxes_xyxy))


class Yolo11Detector(BaseDetector):
    """Camada de inferência YOLO11 com saída padronizada e instrumentação.

    ``detect`` continua retornando ``supervision.Detections`` para preservar o
    contrato com o ByteTrack. ``detect_with_metadata`` expõe também caixas
    normalizadas e tempos de execução por frame.
    """

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
        try:
            self.model = YOLO(str(weights))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Não foi possível abrir os pesos do detector: {weights}."
            ) from error
        except RuntimeError as error:
            raise RuntimeError(
                "Falha ao carregar o YOLO. Verifique torch, torchvision, "
                "Ultralytics e o dispositivo configurado."
            ) from error
        self.config = config
        self.names = self._read_model_names()
        self.last_result: DetectionBatch | None = None

    def detect(self, frame: np.ndarray, frame_idx: int) -> sv.Detections:
        """Detecta nadadores e retorna a estrutura compatível com o ByteTrack."""
        return self.detect_with_metadata(frame, frame_idx).detections

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Valida e padroniza o frame antes da inferência Ultralytics."""
        if frame is None:
            raise ValueError("O frame recebido é None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"O frame deve ser numpy.ndarray, recebido {type(frame)!r}."
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"O frame deve ter shape [altura, largura, 3], recebido {frame.shape}."
            )
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(f"Dimensões inválidas de frame: {frame.shape}.")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def detect_with_metadata(self, frame: np.ndarray, frame_idx: int) -> DetectionBatch:
        """Executa a inferência e retorna caixas, scores, classes e tempos."""
        preprocess_start = time.perf_counter()
        prepared_frame = self.preprocess(frame)
        preprocess_ms = self._elapsed_ms(preprocess_start) if self.config.collect_timing else 0.0

        inference_start = time.perf_counter()
        results = self.model.predict(
            prepared_frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            classes=self.config.classes,
            device=self.config.device,
            verbose=False,
            stream=False,
        )[0]
        inference_ms = self._elapsed_ms(inference_start) if self.config.collect_timing else 0.0

        postprocess_start = time.perf_counter()
        batch = self.postprocess(
            results,
            prepared_frame.shape,
            frame_idx,
            preprocess_ms,
            inference_ms,
        )
        if self.config.collect_timing:
            batch.timings_ms["postprocess"] = self._elapsed_ms(postprocess_start)
            batch.timings_ms["total"] = sum(batch.timings_ms.values())
        self.last_result = batch
        return batch

    def postprocess(
        self,
        results,
        frame_shape: tuple[int, ...],
        frame_id: int,
        preprocess_ms: float,
        inference_ms: float,
    ) -> DetectionBatch:
        """Converte a saída do YOLO para formatos downstream padronizados."""
        frame_height, frame_width = frame_shape[:2]
        empty = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty((0,), dtype=np.float32)
        empty_classes = np.empty((0,), dtype=np.int32)

        if results.boxes is None or len(results.boxes) == 0:
            detections = sv.Detections(
                xyxy=empty,
                confidence=empty_scores,
                class_id=empty_classes,
            )
            return DetectionBatch(
                frame_id, frame_width, frame_height, empty, empty.copy(), empty.copy(),
                empty_scores, empty_classes, detections,
                {"preprocess": preprocess_ms, "inference": inference_ms,
                 "postprocess": 0.0, "total": preprocess_ms + inference_ms},
            )

        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = results.boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = results.boxes.cls.cpu().numpy().astype(np.int32)
        boxes = self._clip_boxes(boxes, frame_width, frame_height)
        valid = self._valid_boxes(boxes)
        boxes, confidences, class_ids = boxes[valid], confidences[valid], class_ids[valid]
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes)
        boxes_normalized = self._normalize_boxes(boxes_cxcywh, frame_width, frame_height)
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )
        detections.data["frame_id"] = np.full(len(boxes), frame_id, dtype=np.int32)
        detections.data["cxcywh_normalized"] = boxes_normalized
        return DetectionBatch(
            frame_id, frame_width, frame_height, boxes, boxes_cxcywh,
            boxes_normalized, confidences, class_ids, detections,
            {"preprocess": preprocess_ms, "inference": inference_ms,
             "postprocess": 0.0, "total": preprocess_ms + inference_ms},
        )

    def detect_video(
        self,
        source: str | int,
        max_frames: int | None = None,
    ) -> Iterator[DetectionBatch]:
        """Processa um arquivo de vídeo ou câmera frame a frame."""
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir a fonte: {source}")
        frame_id = 0
        try:
            while max_frames is None or frame_id < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                yield self.detect_with_metadata(frame, frame_id)
                frame_id += 1
        finally:
            capture.release()

    def detect_images(
        self,
        image_paths: Iterable[str | Path],
    ) -> Iterator[DetectionBatch]:
        """Processa uma sequência de imagens."""
        for frame_id, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")
            yield self.detect_with_metadata(frame, frame_id)

    def _read_model_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        return {index: name for index, name in enumerate(names)} if isinstance(names, list) else dict(names)

    def export_results(self, batch: DetectionBatch, output_path: str | Path) -> None:
        """Exporta as detecções de um frame para JSON."""
        payload = {
            "frame_id": batch.frame_id,
            "frame_width": batch.frame_width,
            "frame_height": batch.frame_height,
            "boxes_xyxy": batch.boxes_xyxy.tolist(),
            "boxes_cxcywh": batch.boxes_cxcywh.tolist(),
            "boxes_normalized": batch.boxes_normalized.tolist(),
            "confidences": batch.confidences.tolist(),
            "class_ids": batch.class_ids.tolist(),
            "timings_ms": batch.timings_ms,
        }
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: sv.Detections,
        class_names: dict[int, str] | None = None,
    ) -> np.ndarray:
        """Desenha caixas opcionalmente, sem acoplar visualização à inferência."""
        annotated = frame.copy()
        names = class_names or {}
        for box, confidence, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
            x1, y1, x2, y2 = [int(value) for value in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{names.get(int(class_id), int(class_id))} {float(confidence):.2f}"
            cv2.putText(
                annotated, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
        return annotated

    def _valid_boxes(self, boxes: np.ndarray) -> np.ndarray:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        return (
            (widths >= self.config.min_box_width_px)
            & (heights >= self.config.min_box_height_px)
        )

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        clipped = boxes.copy()
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, frame_width)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, frame_height)
        return clipped

    @staticmethod
    def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
        converted = np.zeros_like(boxes)
        converted[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        converted[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        converted[:, 2] = boxes[:, 2] - boxes[:, 0]
        converted[:, 3] = boxes[:, 3] - boxes[:, 1]
        return converted

    @staticmethod
    def _normalize_boxes(
        boxes: np.ndarray, frame_width: int, frame_height: int,
    ) -> np.ndarray:
        normalized = boxes.copy()
        normalized[:, [0, 2]] /= max(frame_width, 1)
        normalized[:, [1, 3]] /= max(frame_height, 1)
        return normalized

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)


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
