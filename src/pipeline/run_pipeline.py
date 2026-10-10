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
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

from src.detection.detector import build_detector
from src.evaluation.metrics import compute_hota, compute_map, compute_mot_metrics, compute_stroke_metrics
from src.pose.pose_estimator import build_pose_estimator
from src.pose.stroke_counter import StrokeCounter
from src.tracking.tracker import SwimmerTracker
from src.utils.config import PipelineConfig
from src.utils.lane_detector import LaneCalibration, LaneDetector
from src.utils.roi_filter import (
    draw_roi, filter_detections_by_lanes, filter_detections_by_roi,
    normalized_lanes_to_pixel_polygons, roi_to_pixel_polygon,
)
from src.utils.visualization import draw_frame, draw_lanes


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
        self.lane_detector = LaneDetector(config.lanes)

        self._preds_mot: dict[int, list[tuple[int, float, float, float, float]]] = {}
        self._preds_map: dict[int, list[tuple[float, float, float, float, float]]] = {}
        self._detection_count = 0
        self._filtered_detection_count = 0
        self._track_observation_count = 0
        self._track_ids_seen: set[int] = set()
        self._track_rows: list[dict[str, int | float]] = []
        self._last_keypoints: dict[int, tuple[int, dict[int, tuple[float, float, float]]]] = {}

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
        lane_polygons_px = []
        lane_calibration = LaneCalibration([], 0.0, 0, "none", 0.0)
        if self.config.roi.enabled and self.config.roi.points:
            roi_polygon_px = roi_to_pixel_polygon(
                self.config.roi.points, self.config.video.width, self.config.video.height
            )
        if self.config.roi.lanes:
            lane_polygons_px = normalized_lanes_to_pixel_polygons(
                self.config.roi.lanes, self.config.video.width, self.config.video.height
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

            if (
                self.config.lanes.enabled
                and self.config.lanes.mode == "auto"
                and frame_idx == self.config.lanes.calibration_frame
            ):
                lane_calibration = self.lane_detector.calibrate(frame)
                lane_polygons_px = self.lane_detector.polygons_to_pixels(
                    lane_calibration.polygons,
                    self.config.video.width,
                    self.config.video.height,
                )

            detections = self.detector.detect(frame, frame_idx)
            self._detection_count += len(detections)
            if roi_polygon_px is not None:
                detections = filter_detections_by_roi(detections, roi_polygon_px)
            if lane_polygons_px:
                detections = filter_detections_by_lanes(detections, lane_polygons_px)
            self._filtered_detection_count += len(detections)
            tracked = self.tracker.update(frame, detections, frame_idx)
            self._track_observation_count += len(tracked)
            if tracked.tracker_id is not None:
                self._track_ids_seen.update(int(track_id) for track_id in tracked.tracker_id)

            keypoints_by_track = {}
            if tracked.tracker_id is not None:
                for index, (bbox, tid, conf) in enumerate(
                    zip(tracked.xyxy, tracked.tracker_id, tracked.confidence)
                ):
                    tid = int(tid)
                    lane_id = -1
                    if "lane_id" in tracked.data:
                        lane_id = int(tracked.data["lane_id"][index])
                    observed_kps = None
                    if frame_idx % self.config.pose.inference_interval_frames == 0:
                        observed_kps = self.pose_estimator.estimate(frame, bbox, frame_idx, tid)
                    self.stroke_counter.update(tid, frame_idx, observed_kps)
                    kps = observed_kps
                    if kps:
                        self._last_keypoints[tid] = (frame_idx, kps)
                    elif tid in self._last_keypoints:
                        last_frame, last_kps = self._last_keypoints[tid]
                        age = frame_idx - last_frame
                        if age <= self.config.pose.stale_pose_max_frames:
                            decay = self.config.pose.stale_pose_confidence_decay ** age
                            kps = {
                                keypoint_idx: (x, y, confidence * decay)
                                for keypoint_idx, (x, y, confidence) in last_kps.items()
                            }
                    if kps:
                        keypoints_by_track[tid] = kps

                    x1, y1, x2, y2 = bbox
                    self._track_rows.append({
                        "frame_number": frame_idx + 1,
                        "track_id": tid,
                        "lane_id": lane_id + 1 if lane_id >= 0 else -1,
                        "confidence": float(conf),
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "head_confidence": self._keypoint_quality(kps, [0, 1, 2, 3, 4]),
                        "shoulder_confidence": self._keypoint_quality(kps, [5, 6]),
                        "arm_confidence": self._keypoint_quality(kps, [7, 8, 9, 10]),
                        "pose_observed": int(observed_kps is not None),
                    })
                    self._preds_mot.setdefault(frame_idx, []).append((tid, x1, y1, x2, y2))
                    self._preds_map.setdefault(frame_idx, []).append((x1, y1, x2, y2, float(conf)))

            if writer is not None:
                annotated = draw_frame(
                    frame, tracked, keypoints_by_track, self.stroke_counter.summary()
                )
                if roi_polygon_px is not None:
                    annotated = draw_roi(annotated, roi_polygon_px)
                if lane_polygons_px:
                    annotated = draw_lanes(annotated, lane_polygons_px)
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
            "video": {
                "source": self.config.video.source,
                "fps": self.config.video.fps,
                "width": self.config.video.width,
                "height": self.config.video.height,
            },
            "models": {
                "detector": self.config.detector.weights_path,
                "pose": self.config.pose.weights_path,
            },
            "tracking": {
                "detections_before_filters": self._detection_count,
                "detections_after_filters": self._filtered_detection_count,
                "track_observations": self._track_observation_count,
                "unique_track_ids": len(self._track_ids_seen),
                "persistent_reassociations": self.tracker.persistent_identity.reassociations
                if self.tracker.persistent_identity is not None else 0,
                "new_logical_identities": self.tracker.persistent_identity.new_identities
                if self.tracker.persistent_identity is not None else len(self._track_ids_seen),
            },
            "lanes": {
                "mode": self.config.lanes.mode,
                "enabled": self.config.lanes.enabled,
                "detected_polygons": len(lane_calibration.polygons),
                "line_count": lane_calibration.line_count,
                "water_ratio": round(lane_calibration.water_ratio, 4),
                "orientation": lane_calibration.orientation,
                "confidence": round(lane_calibration.confidence, 4),
            },
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

    @staticmethod
    def _keypoint_quality(
        keypoints: dict[int, tuple[float, float, float]] | None,
        indices: list[int],
    ) -> float:
        if not keypoints:
            return 0.0
        confidences = [keypoints[index][2] for index in indices if index in keypoints]
        return round(float(sum(confidences) / len(confidences)), 4) if confidences else 0.0

    def _evaluate(self) -> dict:
        gt_mot, gt_map = self._load_gt()
        requested = set(self.config.evaluation.metrics)
        scores = {}
        if not requested or any(metric.startswith("mAP") for metric in requested):
            scores.update(compute_map(
                self._preds_map, gt_map,
                iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            ))
        if not requested or "MOTA" in requested or "IDF1" in requested:
            scores.update(compute_mot_metrics(self._preds_mot, gt_mot))
        if not requested or "HOTA" in requested:
            scores.update(compute_hota(self._preds_mot, gt_mot))
        stroke_gt_path = Path(self.config.evaluation.stroke_events_gt_path)
        if stroke_gt_path.exists():
            stroke_gt: dict[int, list[int]] = defaultdict(list)
            with open(stroke_gt_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stroke_gt[int(row["track_id"])].append(int(row["frame_number"]) - 1)
            predicted_events = {
                int(track_id): sorted({
                    event_frame
                    for wrist in ("left_wrist", "right_wrist")
                    for event_frame in counts.get(f"{wrist}_events", [])
                })
                for track_id, counts in self.stroke_counter.summary().items()
            }
            scores.update(compute_stroke_metrics(
                predicted_events, stroke_gt, self.config.evaluation.stroke_tolerance_frames,
            ))
        return scores

    def _save_outputs(self, results: dict, out_dir: Path) -> None:
        if self.config.output.save_metrics_json:
            with open(out_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Keep the exact typed configuration beside the metrics so an
            # experiment can be reproduced after replacing temporary weights.
            with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
                json.dump(dataclasses.asdict(self.config), f, indent=2, ensure_ascii=False)

        if self.config.output.save_stroke_counts_csv:
            with open(out_dir / "stroke_counts.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["track_id", "left_wrist", "right_wrist", "estimated_total_strokes"])
                for tid, counts in results["stroke_counts"].items():
                    writer.writerow([
                        tid, counts.get("left_wrist", 0), counts.get("right_wrist", 0),
                        counts.get("estimated_total_strokes", 0),
                    ])

            with open(out_dir / "stroke_events.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["track_id", "wrist", "frame_idx_zero_based", "frame_number"])
                for tid, counts in results["stroke_counts"].items():
                    for wrist in ("left_wrist", "right_wrist"):
                        for event_frame in counts.get(f"{wrist}_events", []):
                            writer.writerow([tid, wrist, event_frame, event_frame + 1])

        if self.config.output.save_mot_predictions:
            with open(out_dir / "predictions_mot.txt", "w", encoding="utf-8") as f:
                for frame_idx in sorted(self._preds_mot):
                    for track_id, x1, y1, x2, y2 in self._preds_mot[frame_idx]:
                        width, height = x2 - x1, y2 - y1
                        f.write(
                            f"{frame_idx + 1},{track_id},{x1:.3f},{y1:.3f},"
                            f"{width:.3f},{height:.3f},1,-1,-1\n"
                        )

                with open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8") as f:
                    fields = ["frame_number", "track_id", "lane_id", "confidence",
                              "x1", "y1", "x2", "y2", "head_confidence",
                              "shoulder_confidence", "arm_confidence", "pose_observed"]
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(self._track_rows)
