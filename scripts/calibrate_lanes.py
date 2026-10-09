"""Calibrate normalized lane polygons from one video frame.

Usage:
    venv/bin/python scripts/calibrate_lanes.py --video data/raw/video.mp4 \
        --output data/annotations/lanes.json --lanes 8

Click four vertices for each lane in clockwise order. Press r to reset the
current lane, n to finish the current lane, and q to abort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra polígonos de raias.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Não foi possível abrir o vídeo: {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Não foi possível ler o frame {args.frame}.")

    original = frame.copy()
    polygons: list[list[list[float]]] = []
    current: list[tuple[int, int]] = []
    lane_index = 0
    window = "Lane calibration"

    def redraw() -> None:
        canvas = original.copy()
        for index, polygon in enumerate(polygons):
            points = np.asarray(
                [[int(x * original.shape[1]), int(y * original.shape[0])] for x, y in polygon],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [points], True, (0, 200, 0), 2)
            cv2.putText(canvas, f"lane {index + 1}", tuple(points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
        for x, y in current:
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(canvas, f"lane {lane_index + 1}/{args.lanes}: click 4 points | n=next r=reset q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def on_click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(current) < 4:
            current.append((x, y))
            redraw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    redraw()
    while lane_index < args.lanes:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Calibração cancelada.")
        if key == ord("r"):
            current.clear()
            redraw()
        elif key == ord("n") and len(current) == 4:
            width, height = original.shape[1], original.shape[0]
            polygons.append([[round(x / width, 6), round(y / height, 6)] for x, y in current])
            current.clear()
            lane_index += 1
            redraw()

    cv2.destroyAllWindows()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"lanes": polygons}, indent=2), encoding="utf-8")
    print(f"Salvo em {output}. Copie o valor de 'lanes' para roi.lanes.")


if __name__ == "__main__":
    main()
