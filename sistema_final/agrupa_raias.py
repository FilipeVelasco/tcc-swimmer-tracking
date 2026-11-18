import numpy as np

def _u(b, eixo):
    """Centro da caixa no eixo do agrupamento."""
    return (b[0] + b[2]) / 2 if eixo == "x" else (b[1] + b[3]) / 2


def _tam(b, eixo):
    """Tamanho da caixa no eixo do agrupamento."""
    return b[2] - b[0] if eixo == "x" else b[3] - b[1]


def corta_nos_saltos(tracks, eixo, fator=1.5):
    """Corta a trilha onde o centro salta, no eixo das raias, mais que fator x o tamanho da caixa (seria troca de nadador)."""
    pedacos = []
    for tr in tracks.values():
        frames = sorted(tr)
        atual = {frames[0]: tr[frames[0]]}
        for a, b in zip(frames[:-1], frames[1:]):
            larg = max(_tam(tr[a], eixo), 1.0)
            if abs(_u(tr[b], eixo) - _u(tr[a], eixo)) > fator * larg * max(1, b - a) ** 0.5:
                pedacos.append(atual)
                atual = {}
            atual[b] = tr[b]
        pedacos.append(atual)
    return [p for p in pedacos if len(p) >= 3]


def _k_medias(xs, pesos, centros):
    """As iterações do k-médias em 1 dimensão a partir dos centros dados: (raia de cada pedaço, centros)."""
    n = len(centros)
    rot = np.zeros(len(xs), dtype=int)
    for _ in range(50):
        rot = np.argmin(np.abs(xs[:, None] - centros[None, :]), axis=1)
        novos = np.array([np.average(xs[rot == k], weights=pesos[rot == k]) if (rot == k).any() else centros[k]
                          for k in range(n)])
        if np.allclose(novos, centros):
            break
        centros = novos
    return rot, centros


def _sobras(pedacos, rot, n):
    """Quantas caixas de cada pedaço a raia joga fora: as dos quadros em que um pedaço mais longo da mesma raia já tem caixa."""
    sobra = np.zeros(len(pedacos), dtype=int)
    for k in range(n):
        usados = set()
        for i in sorted(np.flatnonzero(rot == k), key=lambda i: len(pedacos[i]), reverse=True):
            sobra[i] = sum(1 for fr in pedacos[i] if fr in usados)
            usados.update(pedacos[i])
    return sobra


def agrupa_em_raias(pedacos, n, eixo):
    """Cada pedaço vai para uma das N raias pela posição mediana no eixo (k-médias em 1 dimensão). Raia que fica vazia
    recebe o pedaço com mais caixas jogadas fora, se isso aumentar o número de caixas aproveitadas."""
    if not pedacos:
        return np.array([], dtype=int), np.zeros(n)
    xs = np.array([np.median([_u(b, eixo) for b in p.values()]) for p in pedacos])
    pesos = np.array([len(p) for p in pedacos], float)
    ordem = np.argsort(xs)
    acumulado = np.cumsum(pesos[ordem]) / pesos.sum()
    centros = np.array([xs[ordem][min(np.searchsorted(acumulado, (k + 0.5) / n), len(xs) - 1)] for k in range(n)])
    rot, centros = _k_medias(xs, pesos, centros)
    for _ in range(n):                                # no máximo uma troca por raia
        vazias = [k for k in range(n) if not (rot == k).any()]
        sobra = _sobras(pedacos, rot, n)
        if not vazias or not sobra.any():
            break
        i = max(range(len(pedacos)), key=lambda j: (sobra[j], len(pedacos[j]), -j))     # no empate, o pedaço mais longo
        tenta = centros.copy()
        tenta[vazias[0]] = xs[i]
        rot2, centros2 = _k_medias(xs, pesos, tenta)
        if _sobras(pedacos, rot2, n).sum() >= sobra.sum():      # a troca só fica se aproveitar mais caixas
            break
        rot, centros = rot2, centros2
    return rot, centros


def cadeias(tracks, n, eixo):
    """(lista de N raias {quadro: caixa} na ordem do eixo, ainda sem interpolar; caixas aproveitadas; caixas nos pedaços)."""
    pedacos = corta_nos_saltos(tracks, eixo)
    rot, centros = agrupa_em_raias(pedacos, n, eixo)
    raias = []
    for k in np.argsort(centros):
        do_k = sorted([p for p, r in zip(pedacos, rot) if r == k], key=len, reverse=True)
        cad = {}
        for p in do_k:                          # o pedaço mais longo tem prioridade em cada quadro
            for fr, b in p.items():
                cad.setdefault(fr, b)
        raias.append(cad)
    return raias, sum(len(c) for c in raias), sum(len(p) for p in pedacos)


def uma_por_raia_auto(tracks, n, interpola, lacuna_max):
    """(lista de N raias {quadro: caixa}, informação da escolha). Agrupa pelos dois eixos e fica com o que aproveita mais
    caixas. interpola é a função que preenche lacunas curtas dentro da raia; lacuna_max, o tamanho máximo dessas lacunas."""
    cand = {eixo: cadeias(tracks, n, eixo) for eixo in ("x", "y")}
    eixo = "x" if cand["x"][1] >= cand["y"][1] else "y"       # empate: o horizontal, que é o comportamento original
    raias = [interpola(cad, lacuna_max) if cad else {} for cad in cand[eixo][0]]
    return raias, dict(eixo=eixo, aproveitadas={e: cand[e][1] for e in cand}, nos_pedacos={e: cand[e][2] for e in cand})
