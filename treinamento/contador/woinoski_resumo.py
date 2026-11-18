import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def le(arq):
    linhas = Path(arq).read_text(encoding="utf-8", errors="replace").splitlines()
    fps, rx, ry, tnf = (float(x) for x in linhas[1].split(","))
    q, y, nad, nado = [], [], [], []
    for l in linhas[4:]:
        p = l.strip().split(",")
        if len(p) < 4:
            continue
        q.append(int(p[0])); y.append(float(p[1])); nad.append(int(p[2])); nado.append(p[3].strip())
    return fps, (rx, ry), np.array(q), np.array(y), np.array(nad, bool), nado


if __name__ == "__main__":
    pasta = Path(sys.argv[1])
    tot = defaultdict(lambda: dict(arqs=0, quadros=0, nadando=0, picos=0, periodos=[]))
    fpss, ress, mistos = defaultdict(int), defaultdict(int), 0
    for arq in sorted(pasta.glob("*_str.txt")):
        fps, res, q, y, nad, nado = le(arq)
        fpss[fps] += 1; ress[res] += 1
        tipos = sorted(set(nado))
        mistos += len(tipos) > 1
        p, _ = find_peaks(np.where(nad, y, 0.0), height=0.9, distance=3)
        t = tot["+".join(tipos)]
        t["arqs"] += 1; t["quadros"] += len(q); t["nadando"] += int(nad.sum()); t["picos"] += len(p)
        if len(p) > 2:
            dp = np.diff(p) / fps
            t["periodos"] += dp[dp < 3].tolist()
    print("fps:", dict(fpss), "| resolução:", dict(ress), "| arquivos com mais de um nado:", mistos)
    for k, t in tot.items():
        per = np.array(t["periodos"] or [np.nan])
        print(f"{k:22s}: {t['arqs']:3d} sub-vídeos, {t['quadros']:6d} quadros ({t['nadando']} nadando), {t['picos']:5d} picos | "
              f"período mediano {np.median(per):.2f} s (p10 {np.percentile(per, 10):.2f}, p90 {np.percentile(per, 90):.2f})")
    if len(sys.argv) > 3:
        import cv2
        c = cv2.VideoCapture(sys.argv[2])
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        print("vídeo:", n, "quadros,", c.get(cv2.CAP_PROP_FPS), "fps,", int(c.get(3)), "x", int(c.get(4)))
        fps, res, q, y, nad, nado = le(pasta / (Path(sys.argv[2]).stem + "_str.txt"))
        fr = []
        for k in range(0, 28, 2):
            c.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = c.read()
            f = cv2.resize(f, (150, 150))
            cv2.putText(f, f"{k} y={y[k]:.2f}", (2, 14), 0, 0.4, (0, 255, 255), 1)
            fr.append(f)
        cv2.imwrite(sys.argv[3], np.vstack([np.hstack(fr[:7]), np.hstack(fr[7:])]))
