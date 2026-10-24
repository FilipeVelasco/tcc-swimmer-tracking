"""Executa deteccao em uma imagem usando um modelo treinado."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--modelo", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0", help="GPU (0, 1...) ou cpu.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Imagem nao encontrada: {args.input}")
    if not args.modelo.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {args.modelo}")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf deve estar entre 0 e 1.")

    result = YOLO(str(args.modelo)).predict(
        source=str(args.input),
        conf=args.conf,
        device=args.device,
        verbose=False,
    )[0]
    image = result.plot()
    output_dir = (
        ROOT
        / "resultados"
        / args.modelo.stem
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / args.input.name

    if not cv2.imwrite(str(output), image):
        raise OSError(f"Nao foi possivel salvar: {output}")

    print(f"Detectados: {len(result.boxes)} nadador(es)")
    print(f"Resultado: {output}")

    if args.show:
        cv2.imshow("Deteccao de nadadores", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
