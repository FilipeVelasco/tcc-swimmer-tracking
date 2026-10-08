"""
Filtro de Região de Interesse (ROI).

Descarta detecções cujo centro cai fora de um polígono definido manualmente
(ex.: a área da piscina), evitando falsos positivos de pessoas no fundo da
cena (treinador, salva-vidas, arquibancada) — muito comum ao usar um detector
genérico de "pessoa" (COCO), que não distingue nadador de qualquer outra
pessoa no vídeo.

O polígono é definido em coordenadas NORMALIZADAS (0.0 a 1.0), relativas à
largura/altura do frame — assim a mesma config funciona independente da
resolução exata do vídeo.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


def roi_to_pixel_polygon(roi_points_normalized: list[list[float]], frame_w: int, frame_h: int) -> np.ndarray:
    return np.array(
        [[x * frame_w, y * frame_h] for x, y in roi_points_normalized], dtype=np.float32
    )


def filter_detections_by_roi(
    detections: sv.Detections,
    roi_polygon_px: np.ndarray,
) -> sv.Detections:
    """Mantém só as detecções cujo centro (cx, cy) está dentro do polígono."""
    if len(detections) == 0 or roi_polygon_px is None:
        return detections

    keep_mask = []
    for x1, y1, x2, y2 in detections.xyxy:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = cv2.pointPolygonTest(roi_polygon_px, (float(cx), float(cy)), False) >= 0
        keep_mask.append(inside)

    keep_mask = np.array(keep_mask, dtype=bool)
    return detections[keep_mask]


def draw_roi(frame: np.ndarray, roi_polygon_px: np.ndarray) -> np.ndarray:
    """Desenha o contorno da ROI no frame (útil para conferir visualmente se está bem posicionada)."""
    annotated = frame.copy()
    pts = roi_polygon_px.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
    return annotated
