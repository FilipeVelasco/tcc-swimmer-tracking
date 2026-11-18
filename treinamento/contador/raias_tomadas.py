import os

import numpy as np

P = "novos_youtube_tomadas_1080"
RAIA_DA_TRILHA = {
    f"{P}_saidas_yolo11m_final__yt_0_livre_fem_t1_01.3-09.3s": {8: 1, 7: 2, 6: 3, 5: 4, 4: 5, 3: 6, 2: 7, 1: 8, 0: 9},
    f"{P}_yt0_yolo11m_final__yt_0_livre_fem_t2_09.3-17.9s": {2: 1, 1: 2, 6: 3, 4: 4, 8: 5, 7: 6, 5: 7, 3: 8, 0: 9},
    f"{P}_yt0_yolo11m_final__yt_0_livre_fem_t3_17.9-25.0s": {2: 1, 0: 2, 6: 3, 4: 4, 8: 5, 7: 6, 5: 7, 1: 8, 3: 9},
    f"{P}_yolo11m_final__yt_1_borboleta_t1_03.1-09.8s": {7: 1, 6: 2, 5: 3, 3: 4, 4: 5, 2: 6, 1: 8},
    f"{P}_yolo11m_final__yt_1_borboleta_t3_18.8-25.0s": {2: 1, 3: 2, 1: 3, 0: 4, 4: 6, 5: 7, 6: 8},
    f"{P}_yt2_yolo11m_final__yt_2_livre_masc_t3_15.6-24.0s": {8: 1, 1: 2, 5: 3, 2: 4, 0: 5, 7: 6, 6: 7, 4: 9},
    f"{P}_saidas_yolo11m_final__yt_2_livre_masc_t1_02.1-07.2s": {7: 2, 8: 3, 6: 4, 5: 5, 4: 6, 2: 7, 1: 8, 0: 9},
    f"{P}_yolo11m_final__yt_3_borboleta_t1_06.6-12.3s": {i: i + 1 for i in range(8)},
    f"{P}_yolo11m_final__yt_3_borboleta_t2_17.5-24.0s": {i: i + 1 for i in range(8)},
}


def pela_altura(d):
    """Regra antiga: [(raia, trilha)] pela ordem das trilhas de cima para baixo (centro vertical mediano da caixa)."""
    L = d["caixa"].shape[0]
    cy = [np.nanmedian((d["caixa"][i, :, 1] + d["caixa"][i, :, 3]) / 2) if not np.isnan(d["caixa"][i, :, 0]).all() else 1e9 for i in range(L)]
    return [(r, int(i)) for r, i in enumerate(np.argsort(cy), 1)]


def raias_da_tomada(prefixo, video, d):
    """[(raia, trilha)] da tomada, em ordem de raia. Tomada fora da tabela: regra antiga, com aviso."""
    chave = f"{prefixo}__{video}"
    if os.environ.get("TCC_RAIAS_ANTIGAS") == "1":
        return pela_altura(d)
    if chave not in RAIA_DA_TRILHA:
        print(f"AVISO: {chave} não está em raias_tomadas.RAIA_DA_TRILHA; raia atribuída pela altura na imagem (não conferida)")
        return pela_altura(d)
    return sorted((r, i) for i, r in RAIA_DA_TRILHA[chave].items())
