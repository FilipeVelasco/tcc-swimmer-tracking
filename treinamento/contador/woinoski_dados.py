import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from woinoski_resumo import le

RAIZ = Path(__file__).resolve().parent.parent
FPS = 15.0
NADO = {"freestyle": 0, "fly": 1, "back": 2, "brest": 3}
base = Path(sys.argv[1])
S, pulados = [], []
for arq in sorted((base / "rotulos" / "textFiles").glob("*_str.txt")):
    video = base / "videos" / (arq.name.replace("_str.txt", ".avi"))
    try:
        fps, res, q, y, nad, nado = le(arq)
    except Exception as e:                      # arquivo de rótulo vazio ou quebrado
        pulados.append((arq.name, repr(e)[:60]))
        continue
    tipos = [t for t in set(nado) if t in NADO]
    if not video.exists() or len(q) < 3 * fps or len(tipos) != 1:
        pulados.append((arq.name, f"video={video.exists()} n={len(q)} tipos={tipos}"))
        continue
    cap = cv2.VideoCapture(str(video))
    quadros = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        quadros.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA))
    n0 = min(len(quadros), len(q))
    if n0 < 3 * fps:
        pulados.append((arq.name, f"quadros lidos {len(quadros)}"))
        continue
    X0, y, nad = np.array(quadros[:n0]), y[:n0], nad[:n0]
    p, _ = find_peaks(np.where(nad, y, 0.0), height=0.9, distance=3)
    n = int(n0 / fps * FPS)
    pos = np.arange(n) * fps / FPS
    i0 = np.clip(pos.astype(int), 0, n0 - 1)
    i1 = np.clip(i0 + 1, 0, n0 - 1)
    w = (pos - i0)[:, None, None]
    X = ((1 - w) * X0[i0] + w * X0[i1]).astype(np.uint8)
    corrida, raia = arq.name.split("_")[:2]
    S.append(dict(X=X, ev=p / fps * FPS, total=float(len(p)), nadando=nad[i0], estilo=NADO[tipos[0]], grupo=f"w{int(corrida):02d}",
                  video=arq.name[:-8], trilha=int(raia), det="woinoski", fps0=fps, n0=n0, ev0=p))
with open(RAIZ / "saidas" / "woinoski_48.pkl", "wb") as f:
    pickle.dump(S, f)
print(f"{len(S)} sub-vídeos, {sum(len(s['X']) for s in S)} quadros a 15 q/s, {int(sum(s['total'] for s in S))} braçadas | pulados: {len(pulados)}")
for k in range(4):
    ss = [s for s in S if s["estilo"] == k]
    print(f"  nado {k}: {len(ss)} sub-vídeos, {int(sum(s['total'] for s in ss))} braçadas, corridas {sorted({s['grupo'] for s in ss})}")
print("pulados:", pulados[:12])
