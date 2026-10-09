"""
Testes automatizados do pipeline. Rodar com:  pytest -v

Cobrem os pontos que mais importam para a validade dos resultados do TCC:
  1) o tracker mantém IDs consistentes ao longo do tempo (não "troca" nadadores)
  2) o fallback de re-ID por aparência resgata uma track após uma oclusão curta
  3) o stroke counter conta corretamente um sinal senoidal conhecido
  4) as métricas de avaliação dão os valores esperados em casos triviais/conhecidos
"""
from __future__ import annotations

import sys
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_hota, compute_map, compute_mot_metrics, compute_stroke_metrics
from src.pose.stroke_counter import StrokeCounter
from src.tracking.tracker import AppearanceReID, LaneAwareIdentity, SwimmerTracker
from src.utils.config import PipelineConfig, StrokeCounterConfig, TrackerConfig


@pytest.fixture(scope="module")
def config() -> PipelineConfig:
    return PipelineConfig.load(str(Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"))


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def _make_frame(w=200, h=100, color=(0, 0, 0)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def test_tracker_keeps_consistent_id_for_smoothly_moving_object(config):
    tracker = SwimmerTracker(config.tracker)
    frame = _make_frame()
    ids_over_time = []
    for i in range(10):
        det = sv.Detections(
            xyxy=np.array([[10.0 + i * 2, 10.0, 40.0 + i * 2, 30.0]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0]),
        )
        tracked = tracker.update(frame, det, i)
        ids_over_time.append(int(tracked.tracker_id[0]))
    assert len(set(ids_over_time)) == 1, "O ID deveria se manter constante para um objeto se movendo suavemente"


def test_appearance_reid_rescues_id_after_short_occlusion():
    reid = AppearanceReID(bins=[8, 8, 8], max_lost_age_frames=45, similarity_threshold=0.75)
    frame = _make_frame(color=(200, 50, 50))  # cor distinta para o "nadador"
    bbox = np.array([10, 10, 40, 30])
    emb = reid.compute_embedding(frame, bbox, reid.bins)

    reid.mark_lost(track_id=7, embedding=emb, frame_idx=10)
    # mesma aparência aparece de novo 5 frames depois -> deve resgatar o ID 7
    rescued = reid.try_rescue(emb, frame_idx=15)
    assert rescued == 7


def test_appearance_reid_does_not_rescue_after_expiry():
    reid = AppearanceReID(bins=[8, 8, 8], max_lost_age_frames=10, similarity_threshold=0.75)
    frame = _make_frame(color=(200, 50, 50))
    bbox = np.array([10, 10, 40, 30])
    emb = reid.compute_embedding(frame, bbox, reid.bins)

    reid.mark_lost(track_id=7, embedding=emb, frame_idx=0)
    rescued = reid.try_rescue(emb, frame_idx=100)  # muito depois do max_lost_age_frames
    assert rescued is None


def test_appearance_embedding_includes_center_region():
    frame = _make_frame(w=100, h=80, color=(30, 80, 160))
    embedding = AppearanceReID.compute_embedding(frame, np.array([10, 10, 50, 50]), [4, 4, 4])
    assert embedding.shape == (2 * 4 * 4 * 4,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_lane_constraint_breaks_impossible_cross_lane_identity():
    identity = LaneAwareIdentity(switch_tolerance_frames=0, max_gap_frames=10, distance_factor=2.5)
    bbox = np.array([10.0, 10.0, 30.0, 30.0])
    first_id = identity.update(raw_id=4, lane_id=0, bbox=bbox, frame_idx=0)
    same_id = identity.update(raw_id=4, lane_id=0, bbox=bbox, frame_idx=1)
    crossed_id = identity.update(raw_id=4, lane_id=1, bbox=bbox, frame_idx=2)
    assert first_id == same_id
    assert crossed_id != first_id


def test_lane_constraint_reassociates_new_backend_id_in_same_lane():
    identity = LaneAwareIdentity(switch_tolerance_frames=0, max_gap_frames=10, distance_factor=2.5)
    bbox = np.array([10.0, 10.0, 30.0, 30.0])
    first_id = identity.update(raw_id=4, lane_id=0, bbox=bbox, frame_idx=0)
    recovered_id = identity.update(raw_id=9, lane_id=0, bbox=bbox + 1, frame_idx=5)
    assert recovered_id == first_id


def test_tracker_uses_independent_backend_per_lane():
    tracker = SwimmerTracker(TrackerConfig(lane_constraint_enabled=True))
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [40, 10, 50, 20]], dtype=np.float32),
        confidence=np.array([0.9, 0.9], dtype=np.float32),
        class_id=np.array([0, 0]),
    )
    detections.data["lane_id"] = np.array([0, 1])
    tracked = tracker.update(_make_frame(w=60, h=60), detections, frame_idx=0)
    assert len(tracked) == 2
    assert tracked.data["lane_id"].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Stroke counting
# ---------------------------------------------------------------------------

def test_stroke_counter_counts_known_sine_wave():
    """4 ciclos completos de senoide em 8s -> espera-se ~4 braçadas por punho."""
    fps = 30
    duration_s = 8
    freq_hz = 0.5
    config = StrokeCounterConfig(
        keypoint_indices=[9], min_confidence=0.1, smoothing_window=3,
        min_peak_distance_sec=0.4, min_peak_prominence=0.05,
    )
    counter = StrokeCounter(config, fps=fps)
    for frame_idx in range(fps * duration_s):
        t = frame_idx / fps
        y = 50 * np.sin(2 * np.pi * freq_hz * t) + 100  # oscila em torno de y=100
        counter.update(track_id=1, frame_idx=frame_idx, keypoints={9: (0.0, y, 0.9)})

    result = counter.count_for_track(1)
    assert 3 <= result["left_wrist"] <= 5, f"Esperado ~4 braçadas, obtido {result['left_wrist']}"


def test_stroke_counter_ignores_low_confidence_keypoints():
    config = StrokeCounterConfig(keypoint_indices=[9], min_confidence=0.5)
    counter = StrokeCounter(config, fps=30)
    for frame_idx in range(30):
        counter.update(track_id=1, frame_idx=frame_idx, keypoints={9: (0.0, float(frame_idx), 0.1)})
    result = counter.count_for_track(1)
    assert result["left_wrist"] == 0, "Keypoints com confiança abaixo do limiar não devem ser contados"


# ---------------------------------------------------------------------------
# Métricas de avaliação
# ---------------------------------------------------------------------------

def test_map_is_one_for_perfect_predictions():
    gt = {0: [(0.0, 0.0, 10.0, 10.0)]}
    preds = {0: [(0.0, 0.0, 10.0, 10.0, 1.0)]}
    result = compute_map(preds, gt, iou_thresholds=[0.5])
    assert result["mAP@0.5"] == pytest.approx(1.0, abs=1e-6)


def test_map_is_zero_for_completely_wrong_predictions():
    gt = {0: [(0.0, 0.0, 10.0, 10.0)]}
    preds = {0: [(100.0, 100.0, 110.0, 110.0, 0.9)]}  # longe do GT, IoU = 0
    result = compute_map(preds, gt, iou_thresholds=[0.5])
    assert result["mAP@0.5"] == pytest.approx(0.0, abs=1e-6)


def test_mota_and_idf1_perfect_match():
    gt = {0: [(1, 0.0, 0.0, 10.0, 10.0)], 1: [(1, 1.0, 0.0, 11.0, 10.0)]}
    preds = {0: [(1, 0.0, 0.0, 10.0, 10.0)], 1: [(1, 1.0, 0.0, 11.0, 10.0)]}
    result = compute_mot_metrics(preds, gt)
    assert result["MOTA"] == pytest.approx(1.0, abs=1e-6)
    assert result["IDF1"] == pytest.approx(1.0, abs=1e-6)


def test_mota_penalizes_id_switch():
    gt = {0: [(1, 0.0, 0.0, 10.0, 10.0)], 1: [(1, 0.0, 0.0, 10.0, 10.0)]}
    # ID trocado de 1 para 2 no segundo frame -> deve penalizar IDF1
    preds = {0: [(1, 0.0, 0.0, 10.0, 10.0)], 1: [(2, 0.0, 0.0, 10.0, 10.0)]}
    result = compute_mot_metrics(preds, gt)
    assert result["IDF1"] < 1.0


def test_hota_is_one_for_perfect_tracking():
    gt = {0: [(1, 0.0, 0.0, 10.0, 10.0)], 1: [(1, 0.0, 0.0, 10.0, 10.0)]}
    preds = {0: [(1, 0.0, 0.0, 10.0, 10.0)], 1: [(1, 0.0, 0.0, 10.0, 10.0)]}
    result = compute_hota(preds, gt)
    assert result["HOTA"] == pytest.approx(1.0, abs=1e-6)


def test_stroke_event_metrics_match_events_with_tolerance():
    result = compute_stroke_metrics(
        predicted_events={1: [10, 30], 2: [50]},
        ground_truth_events={1: [11, 30], 2: [80]},
        tolerance_frames=2,
    )
    assert result["stroke_event_TP"] == 2
    assert result["stroke_event_FP"] == 1
    assert result["stroke_event_FN"] == 1
    assert result["stroke_event_F1"] == pytest.approx(2 / 3)


def test_stroke_event_ground_truth_file_is_loaded(tmp_path, config):
    from src.pipeline.run_pipeline import SwimmerTrackingPipeline

    gt_path = tmp_path / "stroke_events_gt.csv"
    gt_path.write_text("track_id,frame_number\n1,10\n1,30\n", encoding="utf-8")
    run_config = dataclasses.replace(
        config,
        evaluation=dataclasses.replace(
            config.evaluation,
            enabled=True,
            stroke_events_gt_path=str(gt_path),
        ),
    )
    pipeline = SwimmerTrackingPipeline(run_config)
    scores = pipeline._evaluate()
    assert scores["stroke_ground_truth_total"] == 2


def test_config_validation_rejects_invalid_roi(config):
    invalid = dataclasses.replace(
        config,
        roi=dataclasses.replace(config.roi, enabled=True, points=[[0.0, 0.0], [1.2, 0.0]]),
    )
    with pytest.raises(ValueError, match="roi.points"):
        invalid.validate_inputs(require_video=False, require_detector_weights=False,
                               require_pose_weights=False)


def test_config_validation_rejects_missing_real_video(config, tmp_path):
    invalid = dataclasses.replace(
        config,
        mode="real",
        video=dataclasses.replace(config.video, source=str(tmp_path / "missing.mp4")),
    )
    with pytest.raises(ValueError, match="vídeo não encontrado"):
        invalid.validate_inputs(require_detector_weights=False, require_pose_weights=False)


# ---------------------------------------------------------------------------
# Integração: pipeline completo de ponta a ponta (modo sintético)
# ---------------------------------------------------------------------------

def test_full_pipeline_runs_end_to_end_and_produces_sane_results(config, tmp_path):
    """
    Roda o SwimmerTrackingPipeline inteiro (vídeo -> detecção -> tracking ->
    pose -> stroke count -> avaliação) em modo sintético e valida que:
      - todos os 6 nadadores do GT foram rastreados
      - as métricas de avaliação saem em faixas plausíveis (não None, não NaN, 0<=x<=1)
      - a contagem de braçadas de cada nadador é > 0 e compatível com a ordem de grandeza esperada
      - os arquivos de saída (vídeo anotado, json, csv) são realmente criados
    """
    import dataclasses

    from src.pipeline.run_pipeline import SwimmerTrackingPipeline

    out_dir = tmp_path / "outputs"
    run_config = dataclasses.replace(config, output=dataclasses.replace(config.output, dir=str(out_dir)))

    pipeline = SwimmerTrackingPipeline(run_config)
    results = pipeline.run()

    assert results["frames_processed"] > 0
    assert len(results["stroke_counts"]) == 6, "Esperado rastrear os 6 nadadores do GT sintético"
    for tid, counts in results["stroke_counts"].items():
        assert counts["estimated_total_strokes"] > 0, f"Nadador {tid} deveria ter braçadas contadas"

    metrics = results["metrics"]
    for key in ("mAP@0.5", "MOTA", "IDF1", "HOTA"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0, f"{key} fora da faixa [0,1]: {metrics[key]}"
    # com o detector sintético (ruído leve), esperamos métricas altas mas não perfeitas
    assert metrics["mAP@0.5"] > 0.7
    assert metrics["MOTA"] > 0.7

    assert (out_dir / "annotated_output.mp4").exists()
    assert (out_dir / "results.json").exists()
    assert (out_dir / "stroke_counts.csv").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
