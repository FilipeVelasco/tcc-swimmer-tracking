
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bancada_contador as B
import raias_tomadas as RT
import respingo_cabeca as RC

M = Path(__file__).resolve().parents[2] / "marcacoes_braçadas"       # as marcações, na raiz do repositório
FPS = 15.0
GIRA = {"cima": 0, "esquerda": 3, "baixo": 2, "direita": 1, "-": 0}   # np.rot90 k (anti-horário) para a cabeça ir para cima


def amostra(d, i, video, grupo, estilo, tempos=None, total=None, det="m_final"):
    """tempos: instantes (s, relativos ao início do vídeo do npz) das braçadas desta trilha, ou None."""
    fps = float(d["fps"])
    tem = ~np.isnan(d["janela"][i, :, 0])
    idx = np.flatnonzero(tem)
    if len(idx) < 2 * fps:
        return None
    a, z = idx[0], idx[-1] + 1
    cab = RC.sinal_cabeca(d, i, "jan", 150, 0.5, video)[2]
    if "_c48" in d:                       # recortes 48x48: corpo na horizontal, sem depender do sentido do nado
        X = np.asarray(d["_c48"][i, a:z])
        if np.nanmedian(d["janela"][i, :, 3]) > np.nanmedian(d["janela"][i, :, 2]):
            X = np.rot90(X, 1, axes=(1, 2))
    else:
        X = np.rot90(d["cinza_jan"][i, a:z], GIRA[cab], axes=(1, 2))
    n = int((z - a) / fps * FPS)
    pos = np.arange(n) * fps / FPS
    i0 = np.clip(pos.astype(int), 0, len(X) - 1)
    i1 = np.clip(i0 + 1, 0, len(X) - 1)
    w = (pos - i0)[:, None, None]
    X = ((1 - w) * X[i0] + w * X[i1]).astype(np.uint8)
    ev = None
    if tempos is not None:
        ev = (np.asarray(tempos, float) - a / fps) * FPS
        if total is None:
            total = len(tempos)
        ev = ev[(ev >= 0) & (ev <= n - 1)]
    return dict(X=X, ev=ev, total=float(total), grupo=grupo, estilo=estilo, video=video, trilha=i, det=det, cabeca=cab)


USA48 = False


def com48(d, medida, video):
    p = B.RAIZ / "saidas" / "recortes48" / f"{medida}__{video}.npy"
    if USA48:
        assert p.exists(), p
        d["_c48"] = np.load(p, mmap_mode="r")
    return d


CORRECOES = M / "bracadas_correcoes_14-11.csv"     # braçadas acrescentadas depois da conferência com o João: arquivo, raia, tempo_s, origem


def marcas(csv_nome, corrige=True):
    """{raia: [instantes]} do arquivo de marcação; com corrige=True soma as braçadas do arquivo de correções (os CSV
    originais do João não são alterados)."""
    m = defaultdict(list)
    with open(M / csv_nome, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            m[int(l["raia"])].append(float(l["tempo_s"]))
    if corrige and CORRECOES.exists():
        with open(CORRECOES, newline="", encoding="utf-8") as f:
            for l in csv.DictReader(f):
                if l["arquivo"] == csv_nome:
                    m[int(l["raia"])].append(float(l["tempo_s"]))
    return m


# Tomadas com marcas do João que NÃO fazem parte da avaliação (os 78 nadadores ficam os mesmos): entram só no treino,
# com det = "extra". São as duas saídas (mergulho, trecho submerso e primeiras braçadas), rastreadas com 9 raias a partir
# das detecções já guardadas (medida novos_youtube_tomadas_1080_saidas_yolo11m_final). Início de cada tomada no tempo do
# vídeo original: medido por comparação de imagem com o vídeo cortado (inicio_tomadas.py; o deslocamento corte → original
# bate em 1 ms nas tomadas cujo início já era conhecido).
EXTRAS = (
    ("bracadas_yt_0_9raias.csv", "novos_youtube_tomadas_1080_saidas_yolo11m_final", "yt_0_livre_fem_t1_01.3-09.3s", 26.326, "yt_0", 0),
    ("bracadas_yt_2_9raias.csv", "novos_youtube_tomadas_1080_saidas_yolo11m_final", "yt_2_livre_masc_t1_02.1-07.2s", 20.176, "yt_2", 0),
)


def yt(csv_nome, prefixo, tomadas, grupo, estilo, det="m_final"):
    m, out = marcas(csv_nome), []
    for video, t0 in tomadas.items():
        d = com48(dict(np.load(B.REC / f"{prefixo}__{video}.npz")), prefixo, video)
        fps = float(d["fps"])
        L, n = d["caixa"].shape[:2]
        for r, i in RT.raias_da_tomada(prefixo, video, d):          # raia conferida de cada trilha (raias_tomadas.py)
            if (~np.isnan(d["caixa"][i, :, 0])).sum() < 2 * fps:
                continue
            t = np.array(m[r]) - t0
            t = t[(t >= 0) & (t <= n / fps)]
            s = amostra(d, int(i), video, grupo, estilo, tempos=t, total=len(t), det=det)
            if s:
                s["raia"] = r
                out.append(s)
    return out


def extras():
    S = []
    for csv_nome, prefixo, video, t0, grupo, estilo in EXTRAS:
        S += yt(csv_nome, prefixo, {video: t0}, grupo, estilo, det="extra")
    return S


def tudo(com_extras=False):
    P = "novos_youtube_tomadas_1080"
    S = extras() if com_extras else []
    S += yt("bracadas_yt_0_9raias.csv", f"{P}_yt0_yolo11m_final", {"yt_0_livre_fem_t2_09.3-17.9s": 34.368, "yt_0_livre_fem_t3_17.9-25.0s": 42.976}, "yt_0", 0)
    S += yt("bracadas_yt_2_9raias.csv", f"{P}_yt2_yolo11m_final", {"yt_2_livre_masc_t3_15.6-24.0s": 33.69}, "yt_2", 0)
    S += yt("bracadas_yt_1_8raias.csv", f"{P}_yolo11m_final", {"yt_1_borboleta_t1_03.1-09.8s": 9.209, "yt_1_borboleta_t3_18.8-25.0s": 24.825}, "yt_1", 1)
    S += yt("bracadas_yt_3.csv", f"{P}_yolo11m_final", {"yt_3_borboleta_t1_06.6-12.3s": 11.645, "yt_3_borboleta_t2_17.5-24.0s": 22.556}, "yt_3", 1)
    for nome in ("istockphoto-499037978-640_adpp_is", "istockphoto-1326477828-640_adpp_is", "istockphoto-2289313971-640_adpp_is"):
        d = com48(dict(np.load(B.REC / f"novos_istock_1nadador_yolo11m_final__{nome}.npz")), "novos_istock_1nadador_yolo11m_final", nome)
        S.append(amostra(d, 0, nome, "istock", 0, tempos=marcas(f"bracadas_{nome}.csv")[1]))
    # vídeos antigos: instantes marcados quando existe bracadas_<vídeo>.csv (raia i+1 = trilha i, da esquerda para a
    # direita; conferido em confere_ordem_raias.py); o total usado na avaliação continua sendo a contagem original.
    ant = {v: marcas(f"bracadas_{v}.csv") for v in B.VIDEOS if (M / f"bracadas_{v}.csv").exists()}
    for det, dv in B.carrega().items():
        for v in B.VIDEOS:
            com48(dv[v], B.DETECTORES[det], v)
            for i, total in enumerate(B.MANUAL[v]):
                t = None
                if v in ant:                          # vídeo de um nadador: vale qualquer número de raia (o João deixou 8 no 07 e no 08)
                    t = [x for ts in ant[v].values() for x in ts] if len(B.MANUAL[v]) == 1 else ant[v].get(i + 1)
                t = sorted(t) if t else None
                s = amostra(dv[v], i, v, v.replace("natacao_teste_", "ant_"), 0, tempos=t, total=total, det=det)
                if s:
                    S.append(s)
    return S


if __name__ == "__main__":
    USA48 = len(sys.argv) > 1 and sys.argv[1] == "48"       # uso: rede_dados.py 48 <sufixo> [extras]
    S = tudo(com_extras="extras" in sys.argv[3:])
    from collections import Counter
    print(Counter((s["grupo"], s["det"] == "m_final", s["ev"] is not None) for s in S))
    print("quadros:", sum(len(s["X"]) for s in S))
    import pickle
    suf = sys.argv[2] if len(sys.argv) > 2 else ""          # ex.: rede_dados.py 48 _ouro -> rede_dados_48_ouro.pkl
    with open(B.RAIZ / "saidas" / (f"rede_dados_48{suf}.pkl" if USA48 else f"rede_dados{suf}.pkl"), "wb") as f:
        pickle.dump(S, f)
