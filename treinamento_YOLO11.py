"""Treina o modelo YOLO11x para detectar nadadores."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "swimming.v1i.yolov11"
DATA_YAML = DATASET / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="GPU (0, 1...) ou cpu.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch; -1 usa o automatico.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATA_YAML}")

    print(f"Dataset: {DATA_YAML}")
    print("Modelo: yolo11x.pt")
    print(f"Dispositivo: {args.device}")
    print("O yolo11x.pt sera baixado automaticamente se ainda nao existir.")

    model = YOLO("yolo11x.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "resultados"),
        name="treino_yolo11x",
        patience=30,
        cos_lr=True,
        close_mosaic=10,
        cache="disk",
        amp=True,
        plots=True,
        seed=42,
    )


if __name__ == "__main__":
    main()
