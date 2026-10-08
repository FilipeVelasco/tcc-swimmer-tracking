"""
Testes de integração do modo "real" do pipeline (Yolo11Detector / Yolo11PoseEstimator).

Não têm como validar a ACURÁCIA nesse modo sem os pesos treinados de verdade
e sem vídeo real — isso só será possível quando o TCC tiver o servidor e os
dados reais. O objetivo aqui é bem mais restrito e ainda assim valioso:
garantir que o CÓDIGO desses caminhos (carregar modelo, rodar predict,
converter para `supervision.Detections`, recortar bbox e rodar pose) executa
sem erro, usando pesos genéricos do YOLO11/YOLO11-pose como substitutos.
Isso pega bugs de integração ANTES de você ter o modelo treinado em mãos.

Precisam de internet (baixam ~5-6MB do GitHub na primeira vez) — marcados
como `slow` para poderem ser pulados com `pytest -m "not slow"` se necessário.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import build_detector
from src.pose.pose_estimator import build_pose_estimator
from src.utils.config import DetectorConfig, PipelineConfig

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_VIDEO = REPO_ROOT / "data" / "synthetic" / "synthetic_video.mp4"


@pytest.fixture(scope="module")
def sample_frame() -> np.ndarray:
    if not SYNTHETIC_VIDEO.exists():
        pytest.skip("Vídeo sintético não encontrado — rode scripts/generate_synthetic_data.py primeiro.")
    cap = cv2.VideoCapture(str(SYNTHETIC_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip("Não foi possível ler um frame do vídeo sintético.")
    return frame


@pytest.fixture(scope="module")
def generic_detector_weights(tmp_path_factory) -> str:
    """Usa o YOLO11n genérico (pré-treinado no COCO) como substituto dos pesos
    treinados no TCC1 — só para validar o código, não a acurácia."""
    from ultralytics import YOLO

    tmp_dir = tmp_path_factory.mktemp("models")
    weights_path = tmp_dir / "generic_yolo11n.pt"
    YOLO("yolo11n.pt")  # garante o download para o cache do ultralytics
    cached = Path.cwd() / "yolo11n.pt"
    if cached.exists():
        shutil.copy(cached, weights_path)
    else:
        pytest.skip("Não foi possível localizar os pesos genéricos do YOLO11n baixados.")
    return str(weights_path)


def test_yolo11_detector_real_mode_runs_without_error(sample_frame, generic_detector_weights):
    det_cfg = DetectorConfig(weights_path=generic_detector_weights, conf_threshold=0.1, classes=[0])
    detector = build_detector("real", config=det_cfg)
    detections = detector.detect(sample_frame, frame_idx=150)
    # não validamos contagem/acurácia aqui — só que o pipeline de conversão funciona
    assert hasattr(detections, "xyxy")
    assert hasattr(detections, "confidence")


def test_yolo11_detector_real_mode_raises_clear_error_for_missing_weights():
    det_cfg = DetectorConfig(weights_path="models/pesos_que_nao_existem.pt")
    with pytest.raises(FileNotFoundError):
        build_detector("real", config=det_cfg)


def test_yolo11_pose_estimator_real_mode_runs_without_error(sample_frame):
    config = PipelineConfig.load(str(REPO_ROOT / "configs" / "pipeline.yaml"))
    pose_estimator = build_pose_estimator("real", config=config.pose)
    bbox = np.array([500.0, 590.0, 580.0, 625.0])
    result = pose_estimator.estimate(sample_frame, bbox, frame_idx=150, track_id=1)
    # sobre elipses sintéticas o esperado é None (nenhuma pessoa real detectada);
    # o importante é não lançar exceção e devolver o tipo certo quando não-None
    assert result is None or isinstance(result, dict)
