"""
Extrai um frame do vídeo e desenha uma grade com coordenadas NORMALIZADAS
(0.0 a 1.0), para você conseguir estimar visualmente os cantos da área da
piscina e preencher `roi.points` em configs/pipeline_real_bootstrap.yaml —
sem precisar de uma janela interativa (o opencv-python-headless não suporta
cv2.imshow).

Uso:
    python scripts/extract_roi_reference_frame.py --video data/raw/video_natacao.mp4
    python scripts/extract_roi_reference_frame.py --video data/raw/video_natacao.mp4 --frame 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=None,
                         help="Índice do frame a extrair (padrão: frame do meio do vídeo).")
    parser.add_argument("--out", default="outputs_bootstrap/roi_reference_frame.png")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não abriu o vídeo: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = args.frame if args.frame is not None else total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Não consegui ler o frame {frame_idx}.")

    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Grade a cada 0.1 (10%) da largura/altura, com rótulo da coordenada normalizada.
    for i in range(1, 10):
        x = int(w * i / 10)
        y = int(h * i / 10)
        cv2.line(annotated, (x, 0), (x, h), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (0, y), (w, y), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{i/10:.1f}", (x + 3, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{i/10:.1f}", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, annotated)
    print(f"[OK] Frame {frame_idx}/{total} salvo em: {args.out}")
    print(f"Resolução: {w}x{h}")
    print("Abra a imagem, identifique os cantos da área da piscina e leia as")
    print("coordenadas normalizadas (linhas amarelas, rótulos em vermelho) mais")
    print("próximas de cada canto. Preencha em configs/pipeline_real_bootstrap.yaml:")
    print("""
roi:
  enabled: true
  points:
    - [0.0, 0.3]   # <<< ajuste: canto superior-esquerdo da piscina
    - [1.0, 0.3]   # <<< ajuste: canto superior-direito
    - [1.0, 0.9]   # <<< ajuste: canto inferior-direito
    - [0.0, 0.9]   # <<< ajuste: canto inferior-esquerdo
""")


if __name__ == "__main__":
    main()
