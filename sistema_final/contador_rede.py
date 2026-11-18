import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

QPS = 15.0                   # quadros por segundo em que a rede trabalha
LIMIAR = 0.35                # fração da amplitude do sinal exigida de cada pico
NADOS = {"crawl": 0, "borboleta": 1, "costas": 2, "peito": 3}
CICLOS_POR_BRACADA = {"crawl": 0.5, "costas": 0.5, "borboleta": 1.0, "peito": 1.0}   # crawl e costas: cada braço conta uma


class Rede2(nn.Module):
    """Codificador de imagem por quadro (quadro, diferença de 1 e de 2 quadros) + convoluções no tempo -> sinal >= 0.
    nado=True: tem também a saída "está nadando" por quadro (só os pesos treinados com ela)."""

    def __init__(self, T=24, larg=96, nado=False):
        super().__init__()
        c = [nn.Conv2d(3, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.ReLU()] if T >= 48 else [nn.Conv2d(3, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU()]
        c += [nn.Conv2d(24, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(),
              nn.Conv2d(48, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten()]
        self.enc = nn.Sequential(*c)
        self.nado = nn.Embedding(4, 8)
        self.ent = nn.Conv1d(256 + 8, larg, 1)
        self.tmp = nn.ModuleList([nn.Sequential(nn.Conv1d(larg, larg, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(larg), nn.ReLU()) for d in (1, 2, 4, 8, 1, 2)])
        self.sai = nn.Conv1d(larg, 1, 1)
        self.cab_nado = nn.Conv1d(larg, 1, 1) if nado else None
        self.T = T

    def forward(self, x, estilo, com_nado=False):
        B, T = x.shape[:2]
        d = torch.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        d2 = torch.zeros_like(x)
        d2[:, 2:] = x[:, 2:] - x[:, :-2]
        f = self.enc(torch.stack([x, 4 * d, 4 * d2], 2).reshape(B * T, 3, self.T, self.T)).reshape(B, T, -1).transpose(1, 2)
        h = self.ent(torch.cat([f, self.nado(estilo)[:, :, None].expand(B, 8, T)], 1))
        for c in self.tmp:
            h = h + c(h)
        sinal = F.softplus(self.sai(h)[:, 0])
        return (sinal, self.cab_nado(h)[:, 0]) if com_nado else sinal


def carrega_rede(arquivo):
    # weights_only=True: o arquivo só pode trazer tensores e valores simples; um arquivo adulterado não executa código
    ck = torch.load(str(arquivo), map_location="cpu", weights_only=True)
    if ck.get("classe", "Rede2") != "Rede2":
        raise ValueError(f"pesos da classe {ck['classe']}; este arquivo só tem a Rede2")
    rede = Rede2(ck["T"], ck["larg"], nado=bool(ck.get("nado")))
    rede.load_state_dict(ck["estado"])
    rede.eval()
    return rede


def _media_movel(x, k):
    k = max(1, int(k) | 1)
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _mediana_movel(x, k):
    k = max(1, int(k) | 1)
    xp = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def _janela(caixa, fps):
    """cx, cy, w, h suavizados entre a primeira e a última caixa (igual ao contador_respingo)."""
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


def _recortes48(video, janelas):
    """Uma passada no vídeo: recorte em cinza 48x48 de cada janela, em cada quadro."""
    n = max(len(j) for j in janelas.values())
    out = {k: np.zeros((n, 48, 48), np.uint8) for k in janelas}
    cap = cv2.VideoCapture(str(video))
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
            if c - a >= 2 and d - b >= 2:
                out[k][fr] = cv2.resize(gris[b:d, a:c], (48, 48), interpolation=cv2.INTER_AREA)
    cap.release()
    return out


def _entrada(X48, jan, fps, T, camera_lenta=1.0):
    """Recortes da trilha -> sequência a 15 q/s de tempo real, corpo na horizontal, lado T. Devolve (X, primeiro quadro
    do vídeo). camera_lenta = quantas vezes o vídeo está mais lento que o real (1 = velocidade real)."""
    idx = np.flatnonzero(~np.isnan(jan[:, 0]))
    if len(idx) < 2 * fps:
        return None, 0
    a, z = idx[0], idx[-1] + 1
    X = X48[a:z]
    if np.nanmedian(jan[:, 3]) > np.nanmedian(jan[:, 2]):
        X = np.rot90(X, 1, axes=(1, 2))
    n = int((z - a) / fps / camera_lenta * QPS)
    if n < 16:                                         # menos de ~1 s de nado real: não há o que contar
        return None, 0
    pos = np.arange(n) * fps * camera_lenta / QPS
    i0 = np.clip(pos.astype(int), 0, len(X) - 1)
    i1 = np.clip(i0 + 1, 0, len(X) - 1)
    w = (pos - i0)[:, None, None]
    X = ((1 - w) * X[i0] + w * X[i1]).astype(np.uint8)
    if T != 48:
        X = np.stack([cv2.resize(f, (T, T), interpolation=cv2.INTER_AREA) for f in np.ascontiguousarray(X)])
    return np.ascontiguousarray(X), a


def sinal_da_rede(rede, X, nado):
    """X: uint8 [n, T, T] a 15 q/s. Devolve o sinal por quadro."""
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32) / 255)[None]
        return rede(x, torch.tensor([NADOS[nado]]))[0].numpy()


def saidas_da_rede(rede, X, nado):
    """Devolve (sinal por quadro, probabilidade de "está nadando" por quadro ou None se os pesos não têm essa saída)."""
    if rede.cab_nado is None:
        return sinal_da_rede(rede, X, nado), None
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32) / 255)[None]
        sinal, logit = rede(x, torch.tensor([NADOS[nado]]), com_nado=True)
        return sinal[0].numpy(), torch.sigmoid(logit[0]).numpy()


def picos_do_sinal(d, limiar=LIMIAR, intervalo_min=0.0, regular=None, nadando=None, porta=None):
    amp = np.percentile(d, 95) - np.percentile(d, 5)
    p = find_peaks(d, prominence=limiar * amp, distance=max(1, int(round(intervalo_min))))[0]
    if nadando is not None and porta:
        p = p[np.asarray(nadando)[p] >= porta]
    if regular and len(p) >= 4:
        k, r = regular
        iv = np.diff(p)
        perto = iv <= k * np.median(iv)                  # perto[i]: os picos i e i+1 são vizinhos de ritmo
        fica = np.zeros(len(p), bool)
        a = 0
        while a < len(p):
            b = a
            while b < len(p) - 1 and perto[b]:
                b += 1
            if b - a + 1 >= r:
                fica[a:b + 1] = True
            a = b + 1
        if fica.any():
            i = np.flatnonzero(fica)
            p = p[i[0]:i[-1] + 1]
    return p


def ritmo_cv(d, limiar=LIMIAR, nadando=None, porta=None):
    p = find_peaks(d, prominence=limiar * (np.percentile(d, 95) - np.percentile(d, 5)))[0]
    if nadando is not None and porta:
        dentro = np.asarray(nadando) >= porta
        p = p[dentro[p]]
        iv = np.array([b - a for a, b in zip(p[:-1], p[1:]) if dentro[a:b + 1].all()], float)
    else:
        iv = np.diff(p)
    if len(iv) < 3:
        return float("inf")
    return float(iv.std() / iv.mean())


def cascata(bracadas_rede, cv_ritmo, bracadas_respingo, corte=0.20):
    return bracadas_respingo if cv_ritmo > corte else bracadas_rede


def conta_video_rede(video, trilhas, nado, rede, fps=None, limiar=LIMIAR, ciclos_max=None, regular=None, camera_lenta=1.0, porta=None):
    if nado not in NADOS:
        raise ValueError(f"nado deve ser um de {sorted(NADOS)}")
    if camera_lenta <= 0:
        raise ValueError("camera_lenta deve ser maior que zero (1 = velocidade real)")
    if porta and rede.cab_nado is None:
        raise ValueError('porta pedida, mas estes pesos não têm a saída "está nadando"')
    trilhas = {k: np.asarray(v, float) for k, v in trilhas.items()}
    cap = cv2.VideoCapture(str(video))
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    janelas = {k: _janela(c, fps) for k, c in trilhas.items()}
    rec = _recortes48(video, janelas)
    intervalo_min = QPS * 60.0 / ciclos_max * CICLOS_POR_BRACADA[nado] if ciclos_max else 0.0
    res = {}
    for k in trilhas:
        X, a = _entrada(rec[k], janelas[k], fps, rede.T, camera_lenta)
        if X is None:
            res[k] = {"bracadas": 0.0, "tempos_s": [], "sinal": np.zeros(0, np.float32), "inicio_s": 0.0, "cv_ritmo": float("inf")}
            continue
        d, pn = saidas_da_rede(rede, X, nado)
        p = picos_do_sinal(d, limiar, intervalo_min, regular, pn, porta)
        res[k] = {"bracadas": float(len(p)), "tempos_s": [float(a / fps + q * camera_lenta / QPS) for q in p], "sinal": d, "inicio_s": float(a / fps),
                  "cv_ritmo": ritmo_cv(d, limiar, pn, porta)}
        if pn is not None:
            res[k]["nadando"] = pn
    return res
