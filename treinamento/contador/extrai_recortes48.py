"""Recortes em cinza 48x48 da janela suavizada de cada trilha (a mesma janela 'jan' do extrai_recortes, que guarda 24x24).
uso: extrai_recortes48.py <medida> <pasta de vídeos> [filtro]   -> saidas/recortes48/<medida>__<video>.npy  (uint8 [L,n,48,48])"""
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
medida, pasta = sys.argv[1], Path(sys.argv[2])
filtro = sys.argv[3] if len(sys.argv) > 3 else ""
saida = RAIZ / "saidas" / "recortes48"
saida.mkdir(parents=True, exist_ok=True)
for arq in sorted((RAIZ / "saidas" / "recortes").glob(f"{medida}__*.npz")):
    stem = arq.stem.split("__", 1)[1]
    video, dest = pasta / f"{stem}.mp4", saida / f"{arq.stem}.npy"
    if filtro not in stem or not video.exists() or dest.exists():
        continue
    jan = np.load(arq)["janela"]
    L, n = jan.shape[:2]
    out = np.zeros((L, n, 48, 48), np.uint8)
    cap = cv2.VideoCapture(str(video))
    for fr in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gris.shape
        for i in range(L):
            if np.isnan(jan[i, fr, 0]):
                continue
            cx, cy, w, h = jan[i, fr]
            a, b, c, d = max(0, int(cx - w / 2)), max(0, int(cy - h / 2)), min(W, int(cx + w / 2)), min(H, int(cy + h / 2))
            if c - a >= 2 and d - b >= 2:
                out[i, fr] = cv2.resize(gris[b:d, a:c], (48, 48), interpolation=cv2.INTER_AREA)
    np.save(dest, out)
    print(f"{arq.stem}: {L} trilha(s), {n} quadros", flush=True)
