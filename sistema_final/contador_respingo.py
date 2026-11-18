import cv2
import numpy as np
from scipy.signal import find_peaks

T = 24                       # lado do recorte reduzido
CLARO = 150                  # tom de cinza acima do qual o pixel conta como respingo
PARTE = 0.5                  # fração da janela tomada como região da cabeça
PROEMINENCIA = 0.6           # em desvios-padrão do sinal
PASSOS = (0.25, 0.30, 0.35, 0.42, 0.50, 0.60, 0.72, 0.85)   # intervalos mínimos entre braçadas (s)
# Tetos de ritmo: valores de trabalho, ainda sem referência bibliográfica no texto do TCC.
TETO = {"crawl": 2.3, "borboleta": 1.2}                    # braçadas/s (crawl) e ciclos/s (borboleta)


def _media_movel(x, k):
    k = max(1, int(k) | 1)
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _mediana_movel(x, k):
    k = max(1, int(k) | 1)
    xp = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def _janela(caixa, fps):
    """cx, cy, w, h suavizados entre a primeira e a última caixa."""
    n = len(caixa)
    jan = np.full((n, 4), np.nan)
    idx = np.flatnonzero(~np.isnan(caixa[:, 0]))
    if len(idx) < 5:
        return jan
    a, z = idx[0], idx[-1]
    q = np.arange(a, z + 1)
    cx = np.interp(q, idx, (caixa[idx, 0] + caixa[idx, 2]) / 2)
    cy = np.interp(q, idx, (caixa[idx, 1] + caixa[idx, 3]) / 2)
    w = np.interp(q, idx, caixa[idx, 2] - caixa[idx, 0])
    h = np.interp(q, idx, caixa[idx, 3] - caixa[idx, 1])
    jan[a:z + 1, 0], jan[a:z + 1, 1] = _media_movel(cx, 1.5 * fps), _media_movel(cy, 1.5 * fps)
    jan[a:z + 1, 2], jan[a:z + 1, 3] = _mediana_movel(w, 3.0 * fps), _mediana_movel(h, 3.0 * fps)
    return jan


def _le_video(video, janelas):
    """Uma passada no vídeo: recortes em cinza (TxT) de cada janela e deslocamento acumulado do fundo."""
    base = np.random.default_rng(0).random((90, 160)).astype(np.float32)
    (sx, _), _ = cv2.phaseCorrelate(base, np.roll(base, 5, axis=1))
    sinal = 1.0 if sx > 0 else -1.0
    n = max(len(j) for j in janelas.values())
    cinza = {k: np.zeros((n, T, T), np.uint8) for k in janelas}
    valido = {k: np.zeros(n, bool) for k in janelas}
    fundo, acum, ant, han = [], np.zeros(2), None, None
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    esc = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 320.0
    for fr in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gris.shape
        for k, jan in janelas.items():
            if fr >= len(jan) or np.isnan(jan[fr, 0]):
                continue
            cx, cy, w, h = jan[fr]
            a, b, c, d = max(0, int(cx - w / 2)), max(0, int(cy - h / 2)), min(W, int(cx + w / 2)), min(H, int(cy + h / 2))
            if c - a < 2 or d - b < 2:
                continue
            cinza[k][fr] = cv2.resize(gris[b:d, a:c], (T, T), interpolation=cv2.INTER_AREA)
            valido[k][fr] = True
        g = cv2.cvtColor(cv2.resize(frame, (320, int(round(frame.shape[0] / esc)))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if han is None:
            han = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        if ant is not None:
            (dx, dy), resp = cv2.phaseCorrelate(ant, g, han)
            if resp > 0.05 and abs(dx) < 40 and abs(dy) < 40:
                acum = acum + sinal * np.array([dx, dy]) * esc
        fundo.append(acum.copy())
        ant = g
    cap.release()
    return cinza, valido, np.array(fundo), fps


def _frente(caixa, jan, fundo):
    """1 = a cabeça está no fim do eixo (direita ou baixo), 0 = no começo, None = sem deslocamento claro."""
    cx, cy = (caixa[:, 0] + caixa[:, 2]) / 2, (caixa[:, 1] + caixa[:, 3]) / 2
    ok = ~np.isnan(cx)
    if len(fundo) == 0 or ok.sum() < 10:
        return None
    n = min(len(fundo), len(cx))
    ok = ok[:n]
    horizontal = np.nanmedian(jan[:, 2]) >= np.nanmedian(jan[:, 3])
    rel = (cx[:n] - fundo[:n, 0]) if horizontal else (cy[:n] - fundo[:n, 1])
    t = np.flatnonzero(ok)
    incl = np.polyfit(t, rel[ok], 1)[0]
    if abs(incl) * len(t) < 0.5 * np.nanmedian(jan[:, 2] if horizontal else jan[:, 3]):
        return None
    return 1 if incl > 0 else 0


def _conta(x, tem, fps, dist):
    """Devolve (contagem com lacunas preenchidas, quadros dos picos)."""
    idx = np.flatnonzero(tem)
    if len(idx) < int(2 * fps):
        return 0.0, np.array([], int)
    a, z = idx[0], idx[-1] + 1
    cheio = np.interp(np.arange(a, z), idx, x[idx])
    y = _media_movel(cheio - _media_movel(cheio, 2.0 * fps), 0.12 * fps)
    ok = tem[a:z]
    dp = np.std(y[ok])
    if dp <= 0:
        return 0.0, np.array([], int)
    p, _ = find_peaks(y, distance=max(1, int(dist * fps)), prominence=PROEMINENCIA * dp)
    p = p[ok[p]]
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
    return n, p + a


def conta_video_media(video, trilhas, nado, fps=None):
    """Média das contagens com limiar fixo e com limiar adaptado. Foi o método escolhido quando a escolha
    é sempre feita sem o vídeo avaliado (erro 1,88 contra 1,97 do limiar fixo, 195 casos). Lê o vídeo duas vezes."""
    a = conta_video(video, trilhas, nado, fps=fps, claro_adaptativo=False)
    b = conta_video(video, trilhas, nado, fps=fps, claro_adaptativo=True)
    # tempos_s: instantes das braçadas da versão de limiar fixo (servem para desenhar a contagem no vídeo;
    # o total oficial é "bracadas", a média das duas versões)
    return {k: {"bracadas": 0.5 * (a[k]["bracadas"] + b[k]["bracadas"]), "bracadas_fixo": a[k]["bracadas"],
                "bracadas_adaptado": b[k]["bracadas"], "cobertura": a[k]["cobertura"], "cabeca": a[k]["cabeca"],
                "tempos_s": a[k]["tempos_s"], "tempos_adaptado_s": b[k]["tempos_s"]} for k in a}


def conta_video(video, trilhas, nado, fps=None, claro_adaptativo=False):
    if nado not in TETO:
        raise ValueError(f"nado deve ser um de {sorted(TETO)}; este contador não foi validado para outros nados")
    trilhas = {k: np.asarray(v, float) for k, v in trilhas.items()}
    cap = cv2.VideoCapture(str(video))
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    janelas = {k: _janela(c, fps) for k, c in trilhas.items()}
    cinza, valido, fundo, _ = _le_video(video, janelas)
    res = {}
    for k, caixa in trilhas.items():
        jan = janelas[k]
        tem = ~np.isnan(caixa[:, 0]) & valido[k][:len(caixa)]
        vazio = {"bracadas": 0.0, "tempos_s": [], "cabeca": "-", "intervalo_minimo_s": PASSOS[0], "cobertura": 0.0}
        if not tem.any():
            res[k] = vazio
            continue
        claro = CLARO
        if claro_adaptativo:
            claro = 0.5 * CLARO + 0.5 * float(np.percentile(cinza[k][:len(caixa)][~np.isnan(caixa[:, 0])], 75))
        P = cinza[k][:len(caixa)].astype(np.float32) > claro
        m = max(1, int(round(PARTE * T)))
        horizontal = np.nanmedian(jan[:, 2]) >= np.nanmedian(jan[:, 3])
        if horizontal:
            lados, nomes = (P[:, :, :m].mean(axis=(1, 2)), P[:, :, T - m:].mean(axis=(1, 2))), ("esquerda", "direita")
        else:
            lados, nomes = (P[:, :m].mean(axis=(1, 2)), P[:, T - m:].mean(axis=(1, 2))), ("cima", "baixo")
        frente = _frente(caixa, jan, fundo)
        if frente is None:
            frente = 0 if lados[0][tem].mean() <= lados[1][tem].mean() else 1
        idx = np.flatnonzero(~np.isnan(caixa[:, 0]))
        dur = (idx[-1] - idx[0] + 1) / fps
        for dist in PASSOS:
            n, p = _conta(lados[frente], tem, fps, dist)
            if n / dur <= TETO[nado]:
                break
        res[k] = {"bracadas": n, "tempos_s": [float(q / fps) for q in p], "cabeca": nomes[frente],
                  "intervalo_minimo_s": dist, "cobertura": float(tem[idx[0]:idx[-1] + 1].mean())}
    return res
