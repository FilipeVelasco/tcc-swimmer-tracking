"""
Avaliação do pipeline contra o ground-truth.

Métricas implementadas (as mesmas documentadas no TCC1):
  - mAP@0.5 e mAP@0.5:0.95   -> qualidade da DETECÇÃO
  - MOTA e IDF1              -> qualidade do TRACKING (via `motmetrics`)
  - HOTA                     -> qualidade do TRACKING (implementação própria,
                                 simplificada e single-class, já que bibliotecas
                                 de referência como o TrackEval são pesadas e vão
                                 além do escopo "leve" do projeto; a fórmula
                                 segue Luiten et al., 2020, "HOTA: A Higher Order
                                 Metric for Evaluating Multi-Object Tracking")

Todas as funções trabalham a partir de listas de detecções/tracks por frame,
no formato (frame_idx, id, x1, y1, x2, y2[, confidence]) — o mesmo formato
produzido pelo pipeline e pelo GT sintético (após conversão simples).
"""
from __future__ import annotations

from collections import defaultdict

import motmetrics as mm
import numpy as np


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU entre dois conjuntos de boxes no formato xyxy. Shape: (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ious = np.zeros((len(boxes_a), len(boxes_b)))
    for i, a in enumerate(boxes_a):
        xx1 = np.maximum(a[0], boxes_b[:, 0])
        yy1 = np.maximum(a[1], boxes_b[:, 1])
        xx2 = np.minimum(a[2], boxes_b[:, 2])
        yy2 = np.minimum(a[3], boxes_b[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = area_a[i] + area_b - inter
        ious[i] = np.where(union > 0, inter / union, 0.0)
    return ious


def compute_map(
    predictions_by_frame: dict[int, list[tuple[float, float, float, float, float]]],
    gt_by_frame: dict[int, list[tuple[float, float, float, float]]],
    iou_thresholds: list[float],
) -> dict[str, float]:
    """
    mAP simplificado (single-class): calcula AP em cada limiar de IoU pela
    integração precision-recall (11-point interpolation), depois faz a média.
    Retorna mAP para cada limiar pedido + a média sobre 0.5:0.95 se aplicável.
    """
    all_frames = sorted(set(predictions_by_frame) | set(gt_by_frame))
    n_gt_total = sum(len(gt_by_frame.get(f, [])) for f in all_frames)

    aps = {}
    for thr in iou_thresholds:
        # ordena todas as predições (de todos os frames) por confiança desc.
        scored = []
        for f in all_frames:
            for (x1, y1, x2, y2, conf) in predictions_by_frame.get(f, []):
                scored.append((conf, f, (x1, y1, x2, y2)))
        scored.sort(key=lambda t: -t[0])

        matched_gt: dict[int, set[int]] = defaultdict(set)
        tp = np.zeros(len(scored))
        fp = np.zeros(len(scored))

        for i, (conf, f, box) in enumerate(scored):
            gts = gt_by_frame.get(f, [])
            if not gts:
                fp[i] = 1
                continue
            ious = _iou_matrix(np.array([box]), np.array(gts))[0]
            best_j = int(np.argmax(ious)) if len(ious) else -1
            if best_j >= 0 and ious[best_j] >= thr and best_j not in matched_gt[f]:
                tp[i] = 1
                matched_gt[f].add(best_j)
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt_total if n_gt_total > 0 else np.zeros_like(tp_cum)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
            ap += p / 11
        aps[f"AP@{thr:.2f}"] = float(ap)

    result = dict(aps)
    result["mAP@0.5"] = aps.get("AP@0.50", 0.0)
    if len(iou_thresholds) > 1:
        result["mAP@0.5:0.95"] = float(np.mean(list(aps.values())))
    return result


def compute_mot_metrics(
    predictions_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
) -> dict[str, float]:
    """MOTA e IDF1 via `motmetrics`, no padrão MOTChallenge."""
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(predictions_by_frame) | set(gt_by_frame))

    for f in all_frames:
        gts = gt_by_frame.get(f, [])
        preds = predictions_by_frame.get(f, [])
        gt_ids = [g[0] for g in gts]
        pred_ids = [p[0] for p in preds]
        gt_boxes = np.array([g[1:] for g in gts]) if gts else np.zeros((0, 4))
        pred_boxes = np.array([p[1:] for p in preds]) if preds else np.zeros((0, 4))

        # motmetrics espera distância (1 - IoU); "np.nan" descarta o par
        if len(gt_boxes) and len(pred_boxes):
            iou = _iou_matrix(gt_boxes, pred_boxes)
            dist = 1 - iou
            dist[iou < 0.5] = np.nan
        else:
            dist = np.zeros((len(gt_boxes), len(pred_boxes)))

        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1"], name="acc")
    return {
        "MOTA": float(summary["mota"]["acc"]),
        "IDF1": float(summary["idf1"]["acc"]),
    }


def compute_hota(
    predictions_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    alpha_thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """
    HOTA simplificado (single-class), seguindo a decomposição DetA x AssA:
      - Para cada limiar alpha de IoU, casa detecções<->GT gulosamente por IoU.
      - DetA(alpha) = TP / (TP + FN + FP) no nível de detecção.
      - AssA(alpha) = média, sobre os TPs, da IoU de associação de identidade
        (interseção de frames em que pred_id e gt_id casados coincidem, sobre
        a união dos frames em que qualquer um dos dois aparece casado).
      - HOTA(alpha) = sqrt(DetA(alpha) * AssA(alpha)); HOTA final = média sobre alphas.
    """
    if alpha_thresholds is None:
        alpha_thresholds = np.linspace(0.05, 0.95, 19)

    all_frames = sorted(set(predictions_by_frame) | set(gt_by_frame))

    hota_per_alpha = []
    for alpha in alpha_thresholds:
        tp, fp, fn = 0, 0, 0
        # Para cada par (gt_id, pred_id) que já casou pelo menos uma vez, contamos
        # em quantos frames eles co-ocorrem casados (para a associação).
        co_occurrence: dict[tuple[int, int], int] = defaultdict(int)
        gt_id_frame_count: dict[int, int] = defaultdict(int)
        pred_id_frame_count: dict[int, int] = defaultdict(int)

        for f in all_frames:
            gts = gt_by_frame.get(f, [])
            preds = predictions_by_frame.get(f, [])
            for g in gts:
                gt_id_frame_count[g[0]] += 1
            for p in preds:
                pred_id_frame_count[p[0]] += 1

            if not gts or not preds:
                fn += len(gts)
                fp += len(preds)
                continue

            gt_boxes = np.array([g[1:] for g in gts])
            pred_boxes = np.array([p[1:] for p in preds])
            ious = _iou_matrix(gt_boxes, pred_boxes)

            matched_gt, matched_pred = set(), set()
            # matching guloso por IoU decrescente (suficiente p/ cenário de poucos alvos)
            pairs = [(ious[i, j], i, j) for i in range(len(gts)) for j in range(len(preds))]
            pairs.sort(key=lambda t: -t[0])
            for iou_val, i, j in pairs:
                if iou_val < alpha or i in matched_gt or j in matched_pred:
                    continue
                matched_gt.add(i)
                matched_pred.add(j)
                tp += 1
                co_occurrence[(gts[i][0], preds[j][0])] += 1

            fn += len(gts) - len(matched_gt)
            fp += len(preds) - len(matched_pred)

        det_a = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

        if tp == 0:
            ass_a = 0.0
        else:
            weighted_ass = 0.0
            for (gt_id, pred_id), tpa in co_occurrence.items():
                union = gt_id_frame_count[gt_id] + pred_id_frame_count[pred_id] - tpa
                ass_iou = tpa / union if union > 0 else 0.0
                weighted_ass += tpa * ass_iou
            ass_a = weighted_ass / tp

        hota_per_alpha.append(float(np.sqrt(det_a * ass_a)))

    return {
        "HOTA": float(np.mean(hota_per_alpha)),
        "DetA": det_a,   # valor no último alpha avaliado, informativo
        "AssA": ass_a,
    }
