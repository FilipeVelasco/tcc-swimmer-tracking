"""Troca de câmera e número de nadadores.

acha_cortes: um corte é uma mudança brusca entre dois quadros seguidos, na cor (histograma de matiz e saturação) ou na
forma (o quadro em cinza, bem reduzido), seguida de imagem estável: no quadro seguinte a mudança já é pequena. Pega
também a troca entre dois ângulos da mesma piscina, em que as cores quase não mudam. Não confunde com algo passando na
frente da câmera ou com a espuma da largada, que mudam a imagem por vários quadros seguidos. Não pega transição gradual
(fusão de imagens).

estima_nadadores: quantas caixas o detector acha na maior parte dos quadros. Falha quando um nadador fica escondido
na maior parte do vídeo (longe da câmera, dentro do respingo).
"""
import csv
from collections import Counter

import numpy as np

LIMIAR_COR = 0.35        # distância de Bhattacharyya entre os histogramas (0 = iguais, 1 = nada em comum)
LIMIAR_FORMA = 20.0      # diferença média absoluta do quadro em cinza 64x36 (0 a 255)
VEZES_A_MEDIANA = 4.0    # e pelo menos 4 vezes a mediana do último segundo (câmera em movimento não conta como corte)
ESTAVEL_DEPOIS = 0.5     # no quadro seguinte, cor e forma abaixo da metade dos limiares
TOMADA_MINIMA_S = 2.0    # tomadas mais curtas não são contadas
CONF_NADADORES = 0.25    # confiança mínima da caixa para estimar o número de nadadores


def _medidas(img, cv2):
    h, w = img.shape[:2]
    p = cv2.resize(img, (320, max(1, int(320 * h / w))), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([cv2.cvtColor(p, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cinza = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return cv2.normalize(hist, hist).flatten(), cinza


def acha_cortes(video, fps, cv2):
    """Quadros em que começa uma tomada nova (o quadro 0 não entra)."""
    cap = cv2.VideoCapture(str(video))
    ant, cor, forma = None, [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        m = _medidas(img, cv2)
        cor.append(0.0 if ant is None else float(cv2.compareHist(ant[0], m[0], cv2.HISTCMP_BHATTACHARYYA)))
        forma.append(0.0 if ant is None else float(np.abs(ant[1] - m[1]).mean()))
        ant = m
    cap.release()
    cor, forma = np.array(cor), np.array(forma)
    janela = max(1, int(round(fps)))
    cortes = []
    for i in range(2, len(cor) - 1):
        base = slice(max(0, i - janela), i)
        salto = ((cor[i] > LIMIAR_COR and cor[i] >= VEZES_A_MEDIANA * max(np.median(cor[base]), 1e-3))
                 or (forma[i] > LIMIAR_FORMA and forma[i] >= VEZES_A_MEDIANA * max(np.median(forma[base]), 1e-3)))
        estavel = cor[i + 1] < ESTAVEL_DEPOIS * LIMIAR_COR and forma[i + 1] < ESTAVEL_DEPOIS * LIMIAR_FORMA
        if salto and estavel:
            cortes.append(i)
    return cortes, len(cor)


def separa(video, cortes, n_quadros, fps, destino, cv2):
    """Grava cada tomada num arquivo sem perda (FFV1): os quadros ficam idênticos aos do vídeo original.
    Devolve [(arquivo ou None se curta demais, primeiro quadro, último quadro + 1)]."""
    limites = [0] + list(cortes) + [n_quadros]
    trechos = list(zip(limites[:-1], limites[1:]))
    destino.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = []
    for k, (a, b) in enumerate(trechos, 1):
        curta = (b - a) / fps < TOMADA_MINIMA_S
        arq = None if curta else destino / f"tomada_{k:02d}_{a / fps:05.1f}-{b / fps:05.1f}s.avi"
        out = None if curta else cv2.VideoWriter(str(arq), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
        if out is not None and not out.isOpened():
            raise RuntimeError("o OpenCV desta máquina não grava vídeo FFV1; rode com --tomada-unica e separe as tomadas à mão")
        for _ in range(a, b):
            ok, img = cap.read()
            if not ok:
                break
            if out is not None:
                out.write(img)
        if out is not None:
            out.release()
        saida.append((arq, a, b))
    cap.release()
    return saida


def estima_nadadores(arq_csv, n_quadros):
    """Mediana, entre os quadros, do número de caixas com confiança de pelo menos CONF_NADADORES."""
    por_quadro = Counter()
    with open(arq_csv, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if float(linha["conf"]) >= CONF_NADADORES:
                por_quadro[int(linha["frame"])] += 1
    return int(np.median([por_quadro.get(q, 0) for q in range(n_quadros)])) if n_quadros else 0
