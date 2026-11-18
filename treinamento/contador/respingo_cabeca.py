from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import bancada_contador as B

FUNDO = {}


def frente_por_movimento(d, i, video):
    if video not in FUNDO:
        arq = B.RAIZ / "saidas" / "fundo" / f"{video}.npy"
        FUNDO[video] = np.load(arq) if arq.exists() else None
    fundo = FUNDO[video]
    cx = (d["caixa"][i, :, 0] + d["caixa"][i, :, 2]) / 2
    cy = (d["caixa"][i, :, 1] + d["caixa"][i, :, 3]) / 2
    ok = ~np.isnan(cx)
    if fundo is None or ok.sum() < 10:
        return None
    n = min(len(fundo), len(cx))
    ok = ok[:n]
    horizontal = np.nanmedian(d["janela"][i, :, 2]) >= np.nanmedian(d["janela"][i, :, 3])
    rel = (cx[:n] - fundo[:n, 0]) if horizontal else (cy[:n] - fundo[:n, 1])
    t = np.flatnonzero(ok)
    incl = np.polyfit(t, rel[ok], 1)[0]
    comp = np.nanmedian(d["janela"][i, :, 2] if horizontal else d["janela"][i, :, 3])
    if abs(incl) * len(t) < 0.5 * comp:
        return None
    return 1 if incl > 0 else 0


def sinal_cabeca(d, i, rec, lim, parte, video=None):
    P = d[f"cinza_{rec}"][i].astype(np.float32) > lim
    tem = ~np.isnan(d["caixa"][i, :, 0]) & ~np.isnan(d[f"frac_{rec}"][i])
    T = P.shape[1]
    k = max(1, int(round(parte * T)))
    if not tem.any():
        return np.zeros(len(P)), tem, "-"
    if np.nanmedian(d["janela"][i, :, 2]) >= np.nanmedian(d["janela"][i, :, 3]):
        a, b = P[:, :, :k].mean(axis=(1, 2)), P[:, :, T - k:].mean(axis=(1, 2))
        nomes = ("esquerda", "direita")
    else:
        a, b = P[:, :k].mean(axis=(1, 2)), P[:, T - k:].mean(axis=(1, 2))
        nomes = ("cima", "baixo")
    frente = frente_por_movimento(d, i, video) if video else None
    if frente is None:
        frente = 0 if a[tem].mean() <= b[tem].mean() else 1
    return (a, b)[frente], tem, nomes[frente]


def picos(x, tem, fps, dist, prom):
    """Devolve (quadros dos picos no vídeo, máscara ok do trecho, início do trecho)."""
    idx = np.flatnonzero(tem)
    if len(idx) < int(2 * fps):
        return np.array([], int), None, 0
    a, z = idx[0], idx[-1] + 1
    cheio = np.interp(np.arange(a, z), idx, x[idx])
    y = B.media_movel(cheio - B.media_movel(cheio, 2.0 * fps), 0.12 * fps)
    ok = tem[a:z]
    dp = np.std(y[ok])
    if dp <= 0:
        return np.array([], int), ok, a
    p, _ = find_peaks(y, distance=max(1, int(dist * fps)), prominence=prom * dp)
    return np.array([k for k in p if ok[k]], int) + a, ok, a


def conta(d, i, fps, rec, lim, parte, dist, prom, video=None):
    x, tem, _ = sinal_cabeca(d, i, rec, lim, parte, video)
    p, ok, a = picos(x, tem, fps, dist, prom)
    n = float(len(p))
    if len(p) >= 3:
        inter = np.diff(p)
        ritmo = fps / np.median(inter[inter < 2.5 * np.median(inter)])
        k = 0
        while k < len(ok):
            if not ok[k]:
                j = k
                while j < len(ok) and not ok[j]:
                    j += 1
                if (j - k) / fps >= 1.0:
                    n += ritmo * (j - k) / fps
                k = j
            else:
                k += 1
    return n, p
