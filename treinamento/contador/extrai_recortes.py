import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parent.parent
T = 24

nb = nbformat.read(RAIZ / "rastreamento_usado_nos_recortes.ipynb", as_version=4)
codigo = [c.source for c in nb.cells if c.cell_type == "code"]
import os
os.environ["TCC_BASE"] = str(RAIZ / "base_teste")
os.environ.pop("TCC_SMOKE", None)
os.environ["TCC_REPO"] = str(RAIZ / "repo")

MARCA_A = "                cheio[fr] = cad[a] * (1 - t) + cad[b] * t\n"
MARCA_B = "    cheio[frames[-1]] = cad[frames[-1]]\n    return cheio\n"
cel6 = codigo[6]
assert MARCA_A in cel6 and MARCA_B in cel6
cel6 = cel6.replace(MARCA_A, MARCA_A + "                _interp.add(fr)\n")
cel6 = cel6.replace("    frames = sorted(cad)\n    cheio = {}\n", "    frames = sorted(cad)\n    cheio = {}\n    _interp = set()\n", 1)
cel6 = cel6.replace(MARCA_B, "    cheio[frames[-1]] = cad[frames[-1]]\n    INTERP_LOG.append((cheio, _interp))\n    return cheio\n")
assert "INTERP_LOG.append" in cel6 and "_interp.add(fr)" in cel6 and "_interp = set()" in cel6


def media_movel(x, k):
    k = max(1, int(k) | 1)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def mediana_movel(x, k):
    k = max(1, int(k) | 1)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


for medida in sys.argv[1:]:
    ns = {"display": lambda x: None}
    exec(compile(codigo[0], "celula_0", "exec"), ns)
    ns["RUN_NAME"] = medida
    ns["RUN_DIR"] = ns["OUTPUT_DIR"] / medida
    ns["MODELO_PARA_VIDEOS"] = "nao_usado.pt"
    if os.environ.get("TCC_VIDEOS_NOVOS"):
        pasta_novos = Path(os.environ["TCC_VIDEOS_NOVOS"])
        novos = sorted(p.name for p in pasta_novos.glob("*.mp4"))
        ns["VIDEO_DIR"], ns["VIDEOS_RASTREAMENTO"] = pasta_novos, novos
        ns["NADADORES"] = {v: int(os.environ.get("TCC_NADADORES", "1")) for v in novos}
    exec(compile(codigo[1], "celula_1", "exec"), ns)
    ns["display"] = lambda x: None
    ns["YOLO"] = lambda p: None          # o cache já existe: o detector não é carregado
    ns["INTERP_LOG"] = []
    np, cv2 = ns["np"], ns["cv2"]
    for v in ns["VIDEOS_RASTREAMENTO"]:
        assert (ns["RUN_DIR"] / "cache_deteccoes" / (Path(v).stem + ".csv")).exists(), (medida, v)
    # não sobrescrever a tabela rastreador.csv da medida
    cel = cel6.replace("tabela_rastreador.to_csv(RUN_DIR / 'rastreador.csv', index=False)", "pass")
    exec(compile(cel, "celula_6", "exec"), ns)
    saida = RAIZ / "saidas" / "recortes"
    saida.mkdir(parents=True, exist_ok=True)
    limiar = ns["LIMIAR_NAO_AGUA"]
    for nome in ns["VIDEOS_RASTREAMENTO"]:
        info, raias = ns["INFO_VIDEO"][nome], ns["RAIAS"][nome]
        n, fps = info["n_frames"], info["fps"]
        L = len(raias)
        caixa = np.full((L, n, 4), np.nan)
        observada = np.zeros((L, n), bool)
        janela = np.full((L, n, 4), np.nan)        # cx, cy, w, h suavizados
        for i, cad in enumerate(raias):
            interp = next((s for d, s in ns["INTERP_LOG"] if d is cad), set())
            for fr, b in cad.items():
                if fr < n:
                    caixa[i, fr] = b[:4]
                    observada[i, fr] = fr not in interp
            tem = ~np.isnan(caixa[i, :, 0])
            idx = np.flatnonzero(tem)
            if len(idx) < 5:
                continue
            a, z = idx[0], idx[-1]
            fr_all = np.arange(a, z + 1)
            cx = np.interp(fr_all, idx, (caixa[i, idx, 0] + caixa[i, idx, 2]) / 2)
            cy = np.interp(fr_all, idx, (caixa[i, idx, 1] + caixa[i, idx, 3]) / 2)
            w = np.interp(fr_all, idx, caixa[i, idx, 2] - caixa[i, idx, 0])
            h = np.interp(fr_all, idx, caixa[i, idx, 3] - caixa[i, idx, 1])
            janela[i, a:z + 1, 0] = media_movel(cx, 1.5 * fps)
            janela[i, a:z + 1, 1] = media_movel(cy, 1.5 * fps)
            janela[i, a:z + 1, 2] = mediana_movel(w, 3.0 * fps)
            janela[i, a:z + 1, 3] = mediana_movel(h, 3.0 * fps)
        recs = ("caixa", "jan", "jan14")
        dist = {r: np.zeros((L, n, T, T), np.float16) for r in recs}
        cinza = {r: np.zeros((L, n, T, T), np.uint8) for r in recs}
        frac = {r: np.full((L, n), np.nan) for r in recs}
        sinal_caixa = np.full((L, n), np.nan)
        cap = cv2.VideoCapture(str(ns["VIDEO_DIR"] / nome))
        for fr in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            agua = np.median(lab.reshape(-1, 3), axis=0)
            d_agua = np.linalg.norm(lab - agua, axis=2)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = frame.shape[:2]
            for i in range(L):
                ret = {}
                if not np.isnan(caixa[i, fr, 0]):
                    x1, y1, x2, y2 = caixa[i, fr]
                    ret["caixa"] = (max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2)))
                if not np.isnan(janela[i, fr, 0]):
                    cx, cy, w, h = janela[i, fr]
                    for r, f in (("jan", 1.0), ("jan14", 1.4)):
                        ret[r] = (max(0, int(cx - f * w / 2)), max(0, int(cy - f * h / 2)),
                                  min(W, int(cx + f * w / 2)), min(H, int(cy + f * h / 2)))
                for r, (a, b, c, d) in ret.items():
                    if c - a < 2 or d - b < 2:
                        continue
                    rec = d_agua[b:d, a:c]
                    frac[r][i, fr] = float((rec > limiar).mean())
                    dist[r][i, fr] = cv2.resize(rec, (T, T), interpolation=cv2.INTER_AREA)
                    cinza[r][i, fr] = cv2.resize(gris[b:d, a:c], (T, T), interpolation=cv2.INTER_AREA)
                if "caixa" in ret:
                    sinal_caixa[i, fr] = frac["caixa"][i, fr]
        cap.release()
        dados = {"caixa": caixa, "observada": observada, "janela": janela, "fps": fps, "sinal_caixa": sinal_caixa}
        for r in recs:
            dados[f"frac_{r}"], dados[f"dist_{r}"], dados[f"cinza_{r}"] = frac[r], dist[r], cinza[r]
        np.savez_compressed(saida / f"{medida}__{Path(nome).stem}.npz", **dados)
        print(f"{medida} | {nome}: {L} raia(s), {n} quadros, fps {fps:.2f}, "
              f"com caixa {int((~np.isnan(caixa[:, :, 0])).sum())}, interpoladas {int((~np.isnan(caixa[:, :, 0]) & ~observada).sum())}",
              flush=True)
