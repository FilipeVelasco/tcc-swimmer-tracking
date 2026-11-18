
import csv
import warnings
from collections import defaultdict

import numpy as np
import supervision as sv
import torch

# Parâmetros
IMGSZ = 640
DET_CONF = 0.10
TRACK_ACTIVATION = 0.30
MATCH_THRESHOLD = 0.90
LOST_BUFFER = 90
LACUNA_MAX = 60
CICLOS_MAX = 100
CORTE_RITMO = 0.20
PORTA_NADO = None
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
CORES = [(0, 255, 0), (255, 180, 0), (0, 200, 255), (255, 0, 255), (0, 128, 255), (255, 255, 0), (180, 105, 255), (0, 255, 180)]


def detecta(modelo, frame, conf=0.05):
    r = modelo.predict(frame, conf=conf, iou=0.45, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)


def carrega_cache(csv_path):
    por_frame = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as arq:
        for l in csv.DictReader(arq):
            por_frame[int(l['frame'])].append([float(l[k]) for k in ('x1', 'y1', 'x2', 'y2', 'conf')])
    return por_frame


def suprime_contidas(por_frame, limiar=0.6):
    """Caixa dentro de caixa sobre o mesmo nadador: fica só a de maior confiança."""
    novo = defaultdict(list)
    for fr, ds in por_frame.items():
        mantidas = []
        for d in sorted(ds, key=lambda x: -x[4]):
            area_d = max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])
            duplicada = False
            for m in mantidas:
                iw, ih = min(d[2], m[2]) - max(d[0], m[0]), min(d[3], m[3]) - max(d[1], m[1])
                if iw > 0 and ih > 0 and iw * ih / max(min(area_d, (m[2] - m[0]) * (m[3] - m[1])), 1e-6) > limiar:
                    duplicada = True
                    break
            if not duplicada:
                mantidas.append(d)
        novo[fr] = mantidas
    return novo


def bytetrack(por_frame, n_frames, det_conf, ativacao, casamento, buffer):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bt = sv.ByteTrack(track_activation_threshold=ativacao, lost_track_buffer=buffer,
                          minimum_matching_threshold=casamento, frame_rate=30)
    tracks = defaultdict(dict)
    for fr in range(n_frames):
        linhas_fr = [d for d in por_frame.get(fr, []) if d[4] >= det_conf]
        if linhas_fr:
            arr = np.asarray(linhas_fr, dtype=np.float32)
            det = sv.Detections(xyxy=arr[:, :4], confidence=arr[:, 4], class_id=np.zeros(len(arr), dtype=int))
        else:
            det = sv.Detections.empty()
            det.confidence = np.empty((0,), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            saida = bt.update_with_detections(det)
        if saida.tracker_id is not None:
            for box, tid in zip(saida.xyxy, saida.tracker_id):
                tracks[int(tid)][fr] = box.astype(float)
    return tracks


def interpola(cad, max_gap):
    frames = sorted(cad)
    cheio = {}
    for a, b in zip(frames[:-1], frames[1:]):
        cheio[a] = cad[a]
        if 1 < b - a <= max_gap:
            for fr in range(a + 1, b):
                t = (fr - a) / (b - a)
                cheio[fr] = cad[a] * (1 - t) + cad[b] * t
    cheio[frames[-1]] = cad[frames[-1]]
    return cheio


def centro(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def costura(tracks, max_gap=90, fator_dist=2.0):
    """Junta trilhas que não se sobrepõem no tempo e continuam perto de onde a anterior parou."""
    cadeias = []
    for tid in sorted(tracks, key=lambda t: min(tracks[t])):
        tr = tracks[tid]
        ini = min(tr)
        melhor, melhor_d = None, None
        for i, cad in enumerate(cadeias):
            fim = max(cad)
            gap = ini - fim
            if gap <= 0 or gap > max_gap:
                continue
            diag = max(np.hypot(tr[ini][2] - tr[ini][0], tr[ini][3] - tr[ini][1]), 1.0)
            d = float(np.linalg.norm(centro(tr[ini]) - centro(cad[fim])))
            if d <= fator_dist * diag * (1 + 0.1 * gap) and (melhor_d is None or d < melhor_d):
                melhor, melhor_d = i, d
        if melhor is None:
            cadeias.append(dict(tr))
        else:
            cadeias[melhor].update(tr)
    return cadeias


def trilha_unica(tracks, min_lacuna=30, min_cadeia=10, vel_max=0.15):
    """Vídeo de um nadador: a maior cadeia é a principal; as outras só entram onde ela não existe
    por pelo menos `min_lacuna` quadros e se estiverem no caminho do nadador."""
    cads = sorted(costura(tracks), key=len, reverse=True)
    if not cads:
        return {}
    cad = dict(cads[0])
    for outra in cads[1:]:
        if len(outra) < min_cadeia:
            continue
        frames_c = np.array(sorted(cad))

        def livre(f):
            if f in cad:
                return False
            antes, depois = frames_c[frames_c < f], frames_c[frames_c > f]
            return len(antes) == 0 or len(depois) == 0 or depois[0] - antes[-1] >= min_lacuna

        novos = [f for f in sorted(outra) if livre(f)]
        if len(novos) < min_cadeia:
            continue
        coerente = True
        for f in (novos[0], novos[len(novos) // 2], novos[-1]):
            j = int(frames_c[np.argmin(np.abs(frames_c - f))])
            ref = cad[j]
            diag = max(np.hypot(ref[2] - ref[0], ref[3] - ref[1]), 1.0)
            if np.linalg.norm(centro(outra[f]) - centro(ref)) > diag * (1.5 + vel_max * abs(int(f) - j)):
                coerente = False
                break
        if coerente:
            for f in novos:
                cad[f] = outra[f]
    return cad


def exato(conjunto, n_frames, n):
    c = np.zeros(n_frames, dtype=int)
    for tr in conjunto:
        for fr in tr:
            if fr < n_frames:
                c[fr] += 1
    return round(100 * float((c == n).mean()), 1)
