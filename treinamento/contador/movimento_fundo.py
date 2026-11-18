
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
saida = RAIZ / "saidas" / "fundo"
saida.mkdir(parents=True, exist_ok=True)

# confere o sinal da correlação de fase: imagem deslocada +5 px em x
base = np.random.default_rng(0).random((90, 160)).astype(np.float32)
(sx, sy), _ = cv2.phaseCorrelate(base, np.roll(base, 5, axis=1))
SINAL = 1.0 if sx > 0 else -1.0
print("sinal da correlação de fase:", SINAL, round(sx, 2))

for caminho in sys.argv[1:]:
    cap = cv2.VideoCapture(caminho)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    esc = w / 320.0
    jan = None
    ant, pos, acum = None, [], np.zeros(2)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (320, int(round(f.shape[0] / esc)))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if jan is None:
            jan = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        if ant is not None:
            (dx, dy), resp = cv2.phaseCorrelate(ant, g, jan)
            if resp > 0.05 and abs(dx) < 40 and abs(dy) < 40:
                acum = acum + SINAL * np.array([dx, dy]) * esc
        pos.append(acum.copy())
        ant = g
    cap.release()
    pos = np.array(pos)
    np.save(saida / (Path(caminho).stem + ".npy"), pos)
    print(f"{Path(caminho).name}: {len(pos)} quadros, fundo deslocou ({pos[-1][0]:+.0f}, {pos[-1][1]:+.0f}) px no total")
