"""
Orquestra o pipeline completo:
  vídeo -> detecção -> tracking -> pose por track -> stroke counting
        -> (opcional) avaliação contra GT -> vídeo anotado + relatórios

Funciona tanto em modo "synthetic" (fase atual, hipotética) quanto "real"
(quando os pesos treinados e o vídeo real estiverem prontos) — a única
diferença é a config e os arquivos de entrada; o código é o mesmo.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2

from src.detection.detector import build_detector
from src.evaluation.metrics import compute_hota, compute_map, compute_mot_metrics
from src.pose.pose_estimator import build_pose_estimator
from src.pose.stroke_counter import StrokeCounter
from src.tracking.tracker import SwimmerTracker
from src.utils.config import PipelineConfig
from src.utils.roi_filter import draw_roi, filter_detections_by_roi, roi_to_pixel_polygon
from src.utils.visualization import draw_frame


class SwimmerTrackingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        gt_path = config.evaluation.gt_path if config.mode == "synthetic" else None
        gt_kp_path = "data/annotations/gt_keypoints.csv" if config.mode == "synthetic" else None

        self.detector = build_detector(
            mode=config.mode, config=config.detector, gt_path=gt_path,
            frame_size=(config.video.width, config.video.height),
        )
        self.tracker = SwimmerTracker(config.tracker)
        self.pose_estimator = build_pose_estimator(
            mode=config.mode, config=config.pose, gt_keypoints_path=gt_kp_path,
        )
        self.stroke_counter = StrokeCounter(config.stroke_counter, fps=config.video.fps)

        self._preds_mot: dict[int, list[tuple[int, float, float, float, float]]] = {}
        self._preds_map: dict[int, list[tuple[float, float, float, float, float]]] = {}

    def run(self, max_frames: int | None = None) -> dict:
        cap = cv2.VideoCapture(self.config.video.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {self.config.video.source}")

        # Detecta fps/resolução DIRETO do arquivo de vídeo, em vez de confiar
        # cegamente nos valores digitados na config (evita descompasso entre
        # o que está no YAML e o vídeo real, que gera vídeo anotado quebrado
        # ou contagem de braçadas com timing errado).
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if real_fps and real_fps > 0:
            if abs(real_fps - self.config.video.fps) > 0.5 or real_w != self.config.video.width \
                    or real_h != self.config.video.height:
                print(f"[INFO] Config dizia {self.config.video.width}x{self.config.video.height} "
                      f"@ {self.config.video.fps}fps, mas o vídeo real é {real_w}x{real_h} "
                      f"@ {real_fps:.2f}fps. Usando os valores reais do arquivo.")
            self.config.video.fps = round(real_fps)
            self.config.video.width = real_w
            self.config.video.height = real_h
            self.stroke_counter.fps = self.config.video.fps

        roi_polygon_px = None
        if self.config.roi.enabled and self.config.roi.points:
            roi_polygon_px = roi_to_pixel_polygon(
                self.config.roi.points, self.config.video.width, self.config.video.height
            )

        out_dir = Path(self.config.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        writer = None
        if self.config.output.save_annotated_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_dir / "annotated_output.mp4"), fourcc,
                self.config.video.fps, (self.config.video.width, self.config.video.height),
            )

        frame_idx = 0
        t0 = time.time()
        last_print = t0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            detections = self.detector.detect(frame, frame_idx)
            if roi_polygon_px is not None:
                detections = filter_detections_by_roi(detections, roi_polygon_px)
            tracked = self.tracker.update(frame, detections, frame_idx)

            keypoints_by_track = {}
            if tracked.tracker_id is not None:
                for bbox, tid, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
                    tid = int(tid)
                    kps = self.pose_estimator.estimate(frame, bbox, frame_idx, tid)
                    self.stroke_counter.update(tid, frame_idx, kps)
                    if kps:
                        keypoints_by_track[tid] = kps

                    x1, y1, x2, y2 = bbox
                    self._preds_mot.setdefault(frame_idx, []).append((tid, x1, y1, x2, y2))
                    self._preds_map.setdefault(frame_idx, []).append((x1, y1, x2, y2, float(conf)))

            if writer is not None:
                annotated = draw_frame(
                    frame, tracked, keypoints_by_track, self.stroke_counter.summary()
                )
                if roi_polygon_px is not None:
                    annotated = draw_roi(annotated, roi_polygon_px)
                writer.write(annotated)

            frame_idx += 1

            # Progresso a cada ~3s (por tempo, não por nº de frames, já que a
            # velocidade varia bastante entre modo sintético e modo real/CPU).
            now = time.time()
            if now - last_print >= 3.0:
                elapsed_so_far = now - t0
                fps_so_far = frame_idx / elapsed_so_far if elapsed_so_far > 0 else 0
                if total_frames:
                    pct = 100 * frame_idx / total_frames
                    eta_s = (total_frames - frame_idx) / fps_so_far if fps_so_far > 0 else float("inf")
                    print(f"[PROGRESSO] frame {frame_idx}/{total_frames} ({pct:.1f}%) "
                          f"| {fps_so_far:.2f} fps | ETA ~{eta_s:.0f}s")
                else:
                    print(f"[PROGRESSO] frame {frame_idx} | {fps_so_far:.2f} fps")
                last_print = now

        cap.release()
        if writer is not None:
            writer.release()

        elapsed = time.time() - t0
        results = {
            "frames_processed": frame_idx,
            "elapsed_seconds": round(elapsed, 2),
            "fps_processing": round(frame_idx / elapsed, 2) if elapsed > 0 else None,
            "stroke_counts": self.stroke_counter.summary(),
        }

        if self.config.evaluation.enabled and Path(self.config.evaluation.gt_path).exists():
            results["metrics"] = self._evaluate()
        elif self.config.evaluation.enabled:
            print(f"[AVISO] evaluation.enabled=true mas GT não encontrado em "
                  f"'{self.config.evaluation.gt_path}' — pulando avaliação.")


        self._save_outputs(results, out_dir)
        return results

    def _load_gt(self) -> tuple[dict, dict]:
        gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        with open(self.config.evaluation.gt_path, newline="") as f:
            for row in csv.reader(f):
                frame_idx = int(row[0]) - 1
                tid = int(row[1])
                x, y, w, h = (float(v) for v in row[2:6])
                gt_by_frame.setdefault(frame_idx, []).append((tid, x, y, x + w, y + h))
        gt_map = {
            f: [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in v] for f, v in gt_by_frame.items()
        }
        return gt_by_frame, gt_map

    def _evaluate(self) -> dict:
        gt_mot, gt_map = self._load_gt()
        map_scores = compute_map(self._preds_map, gt_map, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        mot_scores = compute_mot_metrics(self._preds_mot, gt_mot)
        hota_scores = compute_hota(self._preds_mot, gt_mot)
        return {**map_scores, **mot_scores, **hota_scores}

    def _save_outputs(self, results: dict, out_dir: Path) -> None:
        if self.config.output.save_metrics_json:
            with open(out_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if self.config.output.save_stroke_counts_csv:
            with open(out_dir / "stroke_counts.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["track_id", "left_wrist", "right_wrist", "estimated_total_strokes"])
                for tid, counts in results["stroke_counts"].items():
                    writer.writerow([
                        tid, counts.get("left_wrist", 0), counts.get("right_wrist", 0),
                        counts.get("estimated_total_strokes", 0),
                    ])
