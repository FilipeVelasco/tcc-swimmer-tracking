import os

PARES = (("ant_3_c1", "ant_3_c2"), ("ant_4_c1", "ant_4_c3"))
PROVA = {g: "+".join(p) for p in PARES for g in p}
# TCC_GRUPOS=arquivo: as ESCOLHAS voltam ao agrupamento antigo (um grupo por arquivo), o das primeiras rodadas.
# Não muda o treino: lá o protocolo antigo se pede com a opção fora_por=arquivo, para ficar escrito no comando.
POR_ARQUIVO = os.environ.get("TCC_GRUPOS") == "arquivo"


def prova(g):
    """Nome da prova do grupo g nas escolhas: o próprio grupo, ou 'a+b' quando dois arquivos são a mesma prova."""
    return g if POR_ARQUIVO else PROVA.get(g, g)


def da_mesma_prova(g):
    """Grupos que saem do treino junto com g (inclui g)."""
    return set(PROVA.get(g, g).split("+"))


def fora_com(g, juntos=()):
    """Grupos que saem do treino junto com g quando, além das provas, há conjuntos que devem sair inteiros (por exemplo
    todos os vídeos de uma mesma competição): se g, ou um vídeo da prova de g, está num conjunto, sai o conjunto todo,
    com as provas inteiras de cada vídeo dele. Sem conjuntos, é o mesmo que da_mesma_prova(g)."""
    f = {g}
    while True:
        novo = set(f)
        for x in f:
            novo |= da_mesma_prova(x)
        for j in juntos:
            if novo & set(j):
                novo |= set(j)
        if novo == f:
            return f
        f = novo


def fora_do_treino(ck_ou_linha):
    """Grupos que ficaram fora do treino, a partir do campo 'fora' de um arquivo de pesos ou de uma linha de rodada:
    texto 'a' ou 'a+b' (rede2.py), lista (curva_ajuste.py) ou ausente (primeiras rodadas: só o próprio grupo)."""
    f = ck_ou_linha.get("fora")
    if not f:
        return {ck_ou_linha["grupo"]} if "grupo" in ck_ou_linha else set()
    return set(f) if isinstance(f, (list, tuple)) else set(str(f).split("+"))
