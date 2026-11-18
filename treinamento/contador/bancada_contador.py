import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, lombscargle

RAIZ = Path(__file__).resolve().parent.parent
REC = RAIZ / "saidas" / "recortes"
MANUAL = {
    "natacao_teste_07": [9], "natacao_teste_08": [18], "natacao_teste_3_c1": [34], "natacao_teste_3_c2": [34],
    "natacao_teste_4_c1": [9, 6, 8, 5, 7, 8, 6, 5], "natacao_teste_4_c3": [15, 13, 14, 13, 13, 17, 11, 10],
}
VIDEOS = list(MANUAL)
DETECTORES = {
    "drone": "medida_original_drone_1011", "x_filipe": "conferencia_v3",
    "n_comp": "medida_yolo11n_640_por_competicao", "s_comp": "medida_yolo11s_640_por_competicao",
    "m_comp": "medida_yolo11m_640_por_competicao", "s_final": "medida_yolo11s_640_final_14_competicoes",
    "m_final": "medida_yolo11m_640_final_14_competicoes",
    "x_comp": "medida_yolo11x_640_por_competicao", "s960_comp": "medida_yolo11s_960_por_competicao",
}
BANDA = (0.4, 2.4)
LIMIAR = 40


def carrega():
    dados = {}
    for det, pasta in DETECTORES.items():
        arqs = {v: REC / f"{pasta}__{v}.npz" for v in VIDEOS}
        if all(a.exists() for a in arqs.values()):
            dados[det] = {v: dict(np.load(a)) for v, a in arqs.items()}
    return dados


# ---------------------------------------------------------------- sinais
def media_movel(x, k):
    k = max(1, int(k) | 1)
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _mascara(d, i, modo):
    tem = ~np.isnan(d["caixa"][i, :, 0])
    if modo == "caixa":
        return tem
    if modo == "obs":
        return tem & d["observada"][i]
    return ~np.isnan(d["janela"][i, :, 0])          # "tudo": da primeira à última aparição


def _pca(P, fps):
    """Componente principal dos recortes com a periodicidade mais nítida na faixa de braçada."""
    X = P.reshape(len(P), -1).astype(np.float32)
    X = X - np.apply_along_axis(media_movel, 0, X, 2.0 * fps)
    if len(X) < int(3 * fps):
        return X.mean(axis=1)
    U, S, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    melhor, nota = 0, -1.0
    freqs = np.linspace(BANDA[0], BANDA[1], 200)
    t = np.arange(len(X)) / fps
    for k in range(min(4, U.shape[1])):
        p = lombscargle(t, U[:, k] * S[k], 2 * np.pi * freqs)
        n = float(p.max() / (p.sum() + 1e-12)) * float(S[k] / S[0]) ** 0.5
        if n > nota:
            melhor, nota = k, n
    return U[:, melhor] * S[melhor]


def sinal(d, i, tipo, fps):
    """tipo = <medida>:<recorte>:<suporte>; devolve vetor com NaN onde não há sinal."""
    medida, rec, sup = tipo.split(":")
    n = d["caixa"].shape[1]
    base = ~np.isnan(d[f"frac_{rec}"][i])
    m = _mascara(d, i, sup) & base
    x = np.full(n, np.nan)
    if m.sum() < 5:
        return x
    if medida == "frac":
        v = d[f"frac_{rec}"][i]
    elif medida == "media":
        v = d[f"dist_{rec}"][i].astype(np.float32).mean(axis=(1, 2))
    elif medida == "frac30":
        v = (d[f"dist_{rec}"][i].astype(np.float32) > 30).mean(axis=(1, 2))
    elif medida == "frac55":
        v = (d[f"dist_{rec}"][i].astype(np.float32) > 55).mean(axis=(1, 2))
    elif medida == "mov":
        c = d[f"cinza_{rec}"][i].astype(np.float32)
        v = np.r_[0.0, np.abs(np.diff(c, axis=0)).mean(axis=(1, 2))]
    elif medida == "respingo":
        v = (d[f"cinza_{rec}"][i] > 200).mean(axis=(1, 2))
    elif medida == "assim":
        P = (d[f"dist_{rec}"][i].astype(np.float32) > LIMIAR)
        w, h = np.nanmedian(d["janela"][i, :, 2]), np.nanmedian(d["janela"][i, :, 3])
        T = P.shape[1]
        if h >= w:      # corpo na vertical da imagem: compara metade esquerda x direita
            v = P[:, :, : T // 2].mean(axis=(1, 2)) - P[:, :, T // 2:].mean(axis=(1, 2))
        else:
            v = P[:, : T // 2].mean(axis=(1, 2)) - P[:, T // 2:].mean(axis=(1, 2))
    elif medida.startswith("topo"):     # ideia do João: o que aparece ACIMA da linha d'água (parte de cima do recorte)
        frac_linhas = int(medida[4:] or 40) / 100
        P = d[f"dist_{rec}"][i].astype(np.float32) > LIMIAR
        v = P[:, : max(1, int(round(frac_linhas * P.shape[1])))].mean(axis=(1, 2))
    elif medida == "alto":              # borda de cima da caixa em relação ao centro suavizado (braço erguido estica a caixa)
        v = (d["janela"][i, :, 1] - d["caixa"][i, :, 1]) / d["janela"][i, :, 3]
    elif medida == "pca":
        idx = np.flatnonzero(base)
        v = np.full(n, np.nan)
        v[idx] = _pca(d[f"dist_{rec}"][i][idx], fps)
    else:
        raise ValueError(tipo)
    x[m] = np.asarray(v, float)[m]
    return x


# ---------------------------------------------------------------- contagem
def passa_banda(x, fps, banda, ordem=2):
    b, a = butter(ordem, [banda[0] / (fps / 2), min(banda[1] / (fps / 2), 0.99)], btype="band")
    return filtfilt(b, a, x - np.mean(x))


def preenche(x, max_gap):
    ok = ~np.isnan(x)
    idx = np.arange(len(x))
    if ok.sum() < 5:
        return x.copy(), ok
    cheio = np.interp(idx, idx[ok], x[ok])
    valido = ok.copy()
    i = 0
    while i < len(x):
        if not ok[i]:
            j = i
            while j < len(x) and not ok[j]:
                j += 1
            if j - i <= max_gap and i > 0 and j < len(x):
                valido[i:j] = True
            i = j
        else:
            i += 1
    return cheio, valido


def trechos(valido, minimo):
    out, i = [], 0
    while i < len(valido):
        if valido[i]:
            j = i
            while j < len(valido) and valido[j]:
                j += 1
            if j - i >= minimo:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def cruzamentos(y, h=0.0):
    ev, alto = [], False
    lim = h * np.std(y)
    for k, v in enumerate(y):
        if not alto and v > lim:
            alto = True
            ev.append(k)
        elif alto and v < -lim:
            alto = False
    return ev


def f_dominante(x, fps, banda=BANDA):
    """Frequência dominante (Lomb-Scargle) só com as amostras que existem, sem tendência lenta."""
    cheio, valido = preenche(x, int(1.0 * fps))
    ts, ys = [], []
    for a, b in trechos(valido, int(2.0 * fps)):
        y = cheio[a:b] - media_movel(cheio[a:b], 2.5 * fps)
        ts.append(np.arange(a, b) / fps)
        ys.append(y / (np.std(y) + 1e-9))
    if not ts:
        return None
    t, y = np.concatenate(ts), np.concatenate(ys)
    freqs = np.linspace(banda[0], banda[1], 300)
    p = lombscargle(t, y, 2 * np.pi * freqs)
    return float(freqs[p.argmax()])


def portao(y, ev, g):
    """Descarta cruzamentos cuja oscilação (pico a vale até o próximo cruzamento) é pequena
    em relação à mediana das oscilações do trecho (nado parado, deslize, chegada)."""
    if g <= 0 or len(ev) < 3:
        return ev
    lim = ev + [len(y)]
    amp = np.array([np.ptp(y[lim[k]:lim[k + 1]]) for k in range(len(ev))])
    return [e for e, a in zip(ev, amp) if a >= g * np.median(amp)]


def conta(x, fps, metodo="onda", lacuna="original", hist=0.0, banda=BANDA, larg=(0.6, 1.6), corrige=None, gate=0.0,
          seg_min=2.0, ordem=2):
    ok = ~np.isnan(x)
    idx = np.flatnonzero(ok)
    if len(idx) < 5:
        return 0.0, 0.0
    cheio, valido = preenche(x, int(1.0 * fps))
    segs = trechos(valido, int(seg_min * fps))
    f0 = f_dominante(x, fps, banda) if metodo in ("adapt", "fase", "freq") else None
    eventos, contadas = [], 0.0
    for a, b in segs:
        if metodo == "onda":
            y = passa_banda(cheio[a:b], fps, banda, ordem)
            ev = portao(y, cruzamentos(y, hist), gate)
        elif metodo == "adapt":
            y = passa_banda(cheio[a:b], fps, (max(0.2, larg[0] * f0), larg[1] * f0))
            ev = portao(y, cruzamentos(y, hist), gate)
        elif metodo == "fase":
            y = passa_banda(cheio[a:b], fps, (max(0.2, larg[0] * f0), larg[1] * f0))
            ph = np.unwrap(np.angle(hilbert(y)))
            contadas += max(0.0, (ph[-1] - ph[0]) / (2 * np.pi))
            ev = cruzamentos(y, hist)
            eventos += [a + k for k in ev]
            continue
        elif metodo == "freq":
            contadas += f0 * (b - a) / fps
            continue
        else:
            raise ValueError(metodo)
        ev = [a + k for k in ev]
        if corrige == "ritmo" and len(ev) >= 4:
            inter = np.diff(ev)
            med = np.median(inter)
            novo, extra = [ev[0]], 0.0
            for e in ev[1:]:
                gap = e - novo[-1]
                if gap < 0.5 * med:
                    continue                                  # braçada falsa: perto demais da anterior
                if gap > 1.6 * med:
                    extra += round(gap / med) - 1             # braçada perdida
                novo.append(e)
            ev = novo
            contadas += extra
        contadas += len(ev)
        eventos += ev
    # ritmo para as lacunas
    if metodo in ("fase", "freq", "adapt") and f0:
        ritmo = f0
    elif len(eventos) >= 3:
        inter = np.diff(eventos)
        ritmo = fps / np.median(inter[inter < 2.5 * np.median(inter)])
    else:
        return contadas, 0.0
    est = 0.0
    if lacuna == "original":
        i = idx[0]
        while i <= idx[-1]:
            if not ok[i]:
                j = i
                while j <= idx[-1] and not ok[j]:
                    j += 1
                if (j - i) / fps >= 1.0:
                    est += ritmo * (j - i) / fps
                i = j
            else:
                i += 1
    elif lacuna == "tudo":
        coberto = sum(b - a for a, b in segs)
        est = ritmo * max(0, (idx[-1] - idx[0] + 1) - coberto) / fps
    return contadas, float(est)


# ---------------------------------------------------------------- avaliação
def avalia(dados, tipo, fator=1.0, **kw):
    """Devolve {det: {(video, raia): total}}."""
    out = {}
    for det, vids in dados.items():
        out[det] = {}
        for v in VIDEOS:
            d = vids[v]
            fps = float(d["fps"])
            for i in range(len(MANUAL[v])):
                c, e = conta(sinal(d, i, tipo, fps), fps, **kw)
                out[det][(v, i)] = fator * (c + e)
    return out


def resumo(res):
    dets = list(res)
    chaves = [(v, i) for v in VIDEOS for i in range(len(MANUAL[v]))]
    real = np.array([MANUAL[v][i] for v, i in chaves], float)
    M = np.array([[res[d][k] for k in chaves] for d in dets])          # det x nadador
    erro = np.abs(M - real)
    por_det = erro.mean(axis=1)
    por_video = {v: float(erro[:, [j for j, k in enumerate(chaves) if k[0] == v]].mean()) for v in VIDEOS}
    return {"mae_por_det": dict(zip(dets, por_det)), "mae": float(por_det.mean()), "pior": float(por_det.max()),
            "disp": float(M.std(axis=0).mean()), "vies": float((M - real).mean()), "por_video": por_video,
            "pm1": float((erro <= 1).mean() * 100), "pm2": float((erro <= 2).mean() * 100), "M": M, "erro": erro}


def linha(nome, r, dets):
    pv = r["por_video"]
    return (f"{nome:44s} " + " ".join(f"{r['mae_por_det'][d]:5.2f}" for d in dets)
            + f" | {r['mae']:5.2f} {r['pior']:5.2f} {r['disp']:5.2f} {r['vies']:+6.2f} {r['pm1']:4.0f} {r['pm2']:4.0f} | "
            + " ".join(f"{pv[v]:5.2f}" for v in VIDEOS))


def cabecalho(dets):
    return (f"{'variação':44s} " + " ".join(f"{d[:5]:>5s}" for d in dets) + " | média  pior  disp   viés  ±1%  ±2% | "
            + " ".join(f"{v.replace('natacao_teste_', ''):>5s}" for v in VIDEOS))


if __name__ == "__main__":
    dados = carrega()
    dets = list(dados)
    print("detectores:", dets)
    if len(sys.argv) > 1 and sys.argv[1] == "diag":
        for v in VIDEOS:
            for i, real in enumerate(MANUAL[v]):
                print(f"\n{v} raia {i + 1} (real {real})")
                for det in dets:
                    d = dados[det][v]
                    fps = float(d["fps"])
                    x = sinal(d, i, "frac:caixa:caixa", fps)
                    ok = ~np.isnan(x)
                    idx = np.flatnonzero(ok)
                    if len(idx) < 5:
                        print(f"  {det:8s} sem trilha")
                        continue
                    dur = (idx[-1] - idx[0] + 1) / fps
                    c, e = conta(x, fps)
                    f0 = f_dominante(x, fps)
                    fj = f_dominante(sinal(d, i, "frac:jan:tudo", fps), fps)
                    print(f"  {det:8s} caixa {ok.sum():4d}/{len(x)} obs {int((ok & d['observada'][i]).sum()):4d} dur {dur:5.1f}s | "
                          f"onda {c:4.0f}+{e:4.1f} | f0 caixa {f0 if f0 is None else round(f0, 2)} Hz -> {0 if f0 is None else f0 * dur:5.1f} | "
                          f"f0 janela {fj if fj is None else round(fj, 2)} Hz | ritmo real se duração total {real / (len(x) / fps):.2f} Hz")
        sys.exit()
    import csv
    import itertools
    variacoes = [("LINHA DE BASE (v3)", "frac:caixa:caixa", {})]
    # A. recorte e medida, com o contador atual
    for t in ["frac:jan:caixa", "frac:jan14:caixa", "frac:jan:tudo", "frac:jan14:tudo", "frac:caixa:obs", "frac:jan:obs",
              "media:caixa:caixa", "media:jan:caixa", "media:jan14:caixa", "frac30:caixa:caixa", "frac55:caixa:caixa",
              "frac30:jan:caixa", "frac55:jan:caixa", "mov:caixa:caixa", "mov:jan:caixa", "respingo:caixa:caixa",
              "respingo:jan:caixa", "pca:caixa:caixa", "pca:jan:caixa", "pca:jan14:caixa", "pca:jan:tudo", "assim:jan:caixa"]:
        variacoes.append((f"A sinal {t}", t, {}))
    # B. contador, sobre o sinal atual e sobre a janela suavizada
    for t in ["frac:caixa:caixa", "frac:jan:caixa", "media:jan:caixa"]:
        for g in (0.2, 0.3, 0.4, 0.5):
            variacoes.append((f"B portão {g} | {t}", t, {"gate": g}))
        for h in (0.1, 0.25, 0.5):
            variacoes.append((f"B histerese {h} | {t}", t, {"hist": h}))
        variacoes.append((f"B corrige pelo ritmo | {t}", t, {"corrige": "ritmo"}))
        variacoes.append((f"B portão 0.3 + corrige | {t}", t, {"gate": 0.3, "corrige": "ritmo"}))
        variacoes.append((f"B lacuna=tudo | {t}", t, {"lacuna": "tudo"}))
        variacoes.append((f"B lacuna=nada | {t}", t, {"lacuna": "nada"}))
        variacoes.append((f"B trecho mínimo 1 s | {t}", t, {"seg_min": 1.0}))
        for bd in ((0.4, 2.0), (0.4, 3.0), (0.6, 2.4), (0.6, 3.0), (0.8, 3.0), (0.3, 2.4)):
            variacoes.append((f"B banda {bd} | {t}", t, {"banda": bd}))
        variacoes.append((f"B filtro ordem 4 | {t}", t, {"ordem": 4}))
        for met in ("adapt", "fase", "freq"):
            variacoes.append((f"B método {met} | {t}", t, {"metodo": met}))
            variacoes.append((f"B método {met} banda 0.6-2.4 | {t}", t, {"metodo": met, "banda": (0.6, 2.4)}))
    variacoes.append(("C obs + lacuna=tudo", "frac:caixa:obs", {"lacuna": "tudo"}))
    variacoes.append(("C obs + portão 0.3", "frac:caixa:obs", {"gate": 0.3}))
    variacoes.append(("C jan obs + portão 0.3", "frac:jan:obs", {"gate": 0.3}))
    print(cabecalho(dets))
    tabela, guardado = [], {}
    for nome, tipo, kw in variacoes:
        res = avalia(dados, tipo, **kw)
        r = resumo(res)
        guardado[nome] = r
        print(linha(nome, r, dets), flush=True)
        tabela.append({"variacao": nome, **{f"mae_{d}": round(r["mae_por_det"][d], 2) for d in dets},
                       "mae_media": round(r["mae"], 2), "mae_pior": round(r["pior"], 2), "dispersao": round(r["disp"], 2),
                       "vies": round(r["vies"], 2), "pm1_pct": round(r["pm1"]), "pm2_pct": round(r["pm2"]),
                       **{v.replace("natacao_teste_", "erro_"): round(e, 2) for v, e in r["por_video"].items()}})
    # modelo nulo: ritmo fixo x duração da trilha, ritmo ajustado nos outros vídeos
    chaves = [(v, i) for v in VIDEOS for i in range(len(MANUAL[v]))]
    real = np.array([MANUAL[v][i] for v, i in chaves], float)
    nulo = {}
    for det in dets:
        dur = []
        for v, i in chaves:
            idx = np.flatnonzero(~np.isnan(dados[det][v]["caixa"][i, :, 0]))
            dur.append((idx[-1] - idx[0] + 1) / float(dados[det][v]["fps"]) if len(idx) else 0.0)
        dur = np.array(dur)
        nulo[det] = {}
        for v in VIDEOS:
            fora = np.array([k[0] != v for k in chaves])
            ritmo = real[fora].sum() / max(dur[fora].sum(), 1e-9)
            for j, k in enumerate(chaves):
                if k[0] == v:
                    nulo[det][k] = ritmo * dur[j]
    r = resumo(nulo)
    print(linha("NULO: ritmo médio dos outros vídeos x duração", r, dets))
    tabela.append({"variacao": "NULO ritmo fixo x duracao", "mae_media": round(r["mae"], 2), "mae_pior": round(r["pior"], 2),
                   **{f"mae_{d}": round(r["mae_por_det"][d], 2) for d in dets}})
    # escolha de variação deixando um vídeo de fora
    nomes = [n for n, _, _ in variacoes]
    E = np.stack([guardado[n]["erro"] for n in nomes])              # variação x det x nadador
    erros_lovo = np.zeros(E.shape[1:])
    print("\nEscolha deixando um vídeo de fora (a variação é escolhida sem ver o vídeo avaliado):")
    for v in VIDEOS:
        dentro = np.array([k[0] == v for k in chaves])
        melhor = int(E[:, :, ~dentro].mean(axis=(1, 2)).argmin())
        erros_lovo[:, dentro] = E[melhor][:, dentro]
        print(f"  fora {v}: escolhida '{nomes[melhor]}' -> erro no vídeo {E[melhor][:, dentro].mean():.2f} "
              f"(linha de base {E[0][:, dentro].mean():.2f})")
    print(f"  erro médio com escolha honesta: {erros_lovo.mean():.2f} | linha de base {E[0].mean():.2f}")
    destino = RAIZ / "saidas" / "bancada_contador.csv"
    campos = []
    for l in tabela:
        campos += [c for c in l if c not in campos]
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(tabela)
    print("gravado em", destino)
