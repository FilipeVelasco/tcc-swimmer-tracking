"""Overlay visual: bboxes com ID da track, keypoints/esqueleto e contagem de braçadas."""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

_box_annotator = sv.BoxAnnotator(thickness=2)
_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# Conexões do esqueleto COCO-17 (pares de índices de keypoints) para desenhar linhas.
_SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # ombros/braços
    (5, 11), (6, 12), (11, 12),                      # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),          # pernas
    (0, 5), (0, 6),                                   # cabeça-ombros
]
_WRIST_INDICES = {9, 10}


def draw_frame(
    frame: np.ndarray,
    tracked: sv.Detections,
    keypoints_by_track: dict[int, dict[int, tuple[float, float, float]]],
    stroke_counts_by_track: dict[int, dict[str, int | float]],
    min_keypoint_conf: float = 0.2,
) -> np.ndarray:
    annotated = frame.copy()

    if tracked.tracker_id is not None and len(tracked) > 0:
        labels = []
        for tid in tracked.tracker_id:
            strokes = stroke_counts_by_track.get(int(tid), {}).get("estimated_total_strokes", 0)
            # OpenCV (cv2.putText, usado internamente pelo LabelAnnotator) não suporta
            # UTF-8 (acentos/cedilha viram '?'), então o texto do overlay fica em ASCII puro.
            # Acentuação correta é usada em todo o resto (logs, docs, comentários).
            labels.append(f"#{int(tid)} | {strokes} strokes")
        annotated = _box_annotator.annotate(annotated, tracked)
        annotated = _label_annotator.annotate(annotated, tracked, labels=labels)

    for tid, kps in keypoints_by_track.items():
        # Modo sintético: só 2 keypoints (punhos). Modo real (YOLO11-pose): até 17 (COCO).
        # As linhas do esqueleto só são desenhadas quando os keypoints necessários existem.
        for i, j in _SKELETON_EDGES:
            if i not in kps or j not in kps:
                continue
            xi, yi, ci = kps[i]
            xj, yj, cj = kps[j]
            if ci < min_keypoint_conf or cj < min_keypoint_conf:
                continue
            cv2.line(annotated, (int(xi), int(yi)), (int(xj), int(yj)), (180, 180, 180), 1, cv2.LINE_AA)

        for kp_idx, (x, y, conf) in kps.items():
            if conf < min_keypoint_conf:
                continue
            color = (0, 255, 255) if kp_idx in _WRIST_INDICES else (255, 0, 255)
            radius = 4 if kp_idx in _WRIST_INDICES else 2
            cv2.circle(annotated, (int(x), int(y)), radius, color, -1)

    return annotated

