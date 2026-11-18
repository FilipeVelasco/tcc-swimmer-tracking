import argparse
import csv
import hashlib
import importlib.metadata as im
import inspect
import json
import platform
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))          # os módulos ficam ao lado deste arquivo (vale também com python -I)

NADOS_VALIDADOS = ("crawl", "borboleta")
# Pesos com que o sistema foi medido, pelo SHA-1 do arquivo. Outros pesos rodam, com aviso.
# O treino final do detector deixou dois arquivos com os mesmos pesos e SHA-1 diferentes (os 659 registros internos
# são idênticos; só muda o nome da pasta interna). Os dois valem como "os da medida".
DETECTORES_DA_MEDIDA = {
    "8efbe3803b0d9232f4356b1c3388d62a0522ff31": "os da medida (yolo11m_640_final_14_competicoes.pt; é o last.pt e o modelo_final.pt do treino final)",
    "870abf365d3054c941ef42bcf665f129eb682785": "os da medida (best.pt do treino final: mesmos pesos em outro arquivo)",
}
CONTADOR_DA_MEDIDA = "a68ec2226554d402089121daa5e6f046aa701734"      # contador_rede_pesos.pt


def sha1(arq):
    h = hashlib.sha1()
    with open(arq, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def para(mensagem):
    print("ERRO: " + mensagem)
    sys.exit(1)


def argumentos():
    p = argparse.ArgumentParser(description="Detecta, rastreia e conta as braçadas dos nadadores de um vídeo.",
                                epilog="Detalhes no LEIA-ME.md, nesta pasta.")
    p.add_argument("video", help="o vídeo; se tiver troca de câmera, cada tomada é contada separada")
    p.add_argument("--nadadores", type=int, help="quantos nadadores aparecem (um por raia); sem ele, o número é estimado")
    p.add_argument("--nado", choices=NADOS_VALIDADOS, help="o nado do vídeo")
    p.add_argument("--saida", help="pasta de saída (padrão: saidas/<nome do vídeo>, ao lado deste arquivo)")
    p.add_argument("--detector", default=str(AQUI / "yolo11m_640_final_14_competicoes.pt"), help="pesos do detector")
    p.add_argument("--contador", default=str(AQUI / "contador_rede_pesos.pt"), help="pesos da rede de contagem")
    p.add_argument("--sem-video", action="store_true", help="não grava o vídeo anotado")
    p.add_argument("--so-deteccao", action="store_true", help="roda só o detector e para")
    p.add_argument("--refaz-deteccao", action="store_true", help="ignora a detecção guardada")
    p.add_argument("--tomada-unica", action="store_true", help="não procura troca de câmera: o vídeo inteiro é uma tomada")
    return p.parse_args()


def cortes_do_video(video, video_sha1, info, saida, cv2):
    """Quadros em que começa uma tomada nova. Guardado em cortes.json, para não reler o vídeo na próxima vez."""
    import tomadas
    regra = {k: getattr(tomadas, k) for k in ("LIMIAR_COR", "LIMIAR_FORMA", "VEZES_A_MEDIANA", "ESTAVEL_DEPOIS")}
    arq = saida / "cortes.json"
    if arq.is_file():
        reg = json.loads(arq.read_text(encoding="utf-8"))
        if reg.get("video_sha1") == video_sha1 and reg.get("regra") == regra:
            return reg["cortes"]
    cortes, _ = tomadas.acha_cortes(video, info["fps"], cv2)
    arq.write_text(json.dumps({"video_sha1": video_sha1, "regra": regra, "cortes": cortes}, indent=1), encoding="utf-8")
    return cortes


def recorta_deteccao(arq_csv, det, q0, q1, arq_tomada, pasta, chave):
    """Grava, na pasta da tomada, as caixas dos quadros q0 a q1 - 1 renumeradas a partir de 0, com o registro que faz o
    comando reaproveitá-las. A tomada tem os mesmos quadros do vídeo (gravada sem perda), então o detector daria as
    mesmas caixas."""
    pasta.mkdir(parents=True, exist_ok=True)
    destino, caixas = pasta / "deteccoes.csv", 0
    with open(arq_csv, newline="", encoding="utf-8") as f, open(destino, "w", newline="", encoding="utf-8") as g:
        r, w = csv.reader(f), csv.writer(g)
        w.writerow(next(r))
        for linha in r:
            q = int(linha[0])
            if q0 <= q < q1:
                w.writerow([q - q0] + linha[1:])
                caixas += 1
    reg = dict(det, **chave, video_sha1=sha1(arq_tomada), video_bytes=arq_tomada.stat().st_size, quadros=q1 - q0,
               video=arq_tomada.name, quadros_lidos=q1 - q0, caixas=caixas, csv_sha1=sha1(destino),
               origem=f"quadros {q0} a {q1 - 1} da detecção do vídeo inteiro")
    reg.pop("refeita_nesta_execucao", None)
    (pasta / "deteccoes.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def roda_por_tomada(a, video, cortes, info, saida, detector, contador, arq_csv, det, chave, cv2):
    """Separa as tomadas, roda este comando em cada uma (com --tomada-unica e a detecção recortada) e junta as tabelas."""
    import subprocess
    import tomadas
    fps = info["fps"]
    print(f"troca de câmera em {len(cortes)} ponto(s): " + ", ".join(f"{q / fps:.2f} s" for q in cortes)
          + f" | cada tomada é contada separada; as de menos de {tomadas.TOMADA_MINIMA_S:.0f} s ficam de fora", flush=True)
    trechos = tomadas.separa(video, cortes, info["n_frames"], fps, saida / "tomadas", cv2)
    python = ([sys.executable] + (["-I"] if sys.flags.isolated else []) + (["-B"] if sys.flags.dont_write_bytecode else [])
              + (["-X", "utf8"] if sys.flags.utf8_mode else []))
    fixos = {k: chave[k] for k in ("detector_sha1", "imgsz", "rastreamento_sha1")}
    linhas, bracadas, instantes = [], [], []
    for k, (arq, q0, q1) in enumerate(trechos, 1):
        linha = {"tomada": k, "inicio_s": round(q0 / fps, 3), "fim_s": round(q1 / fps, 3), "quadros": q1 - q0}
        if arq is None:
            linhas.append(dict(linha, situacao="curta demais, não contada"))
            continue
        pasta = saida / f"tomada_{k:02d}"
        recorta_deteccao(arq_csv, det, q0, q1, arq, pasta, fixos)
        print(f"\n===== tomada {k} de {len(trechos)}: {q0 / fps:.2f} a {q1 / fps:.2f} s =====", flush=True)
        cmd = python + [str(AQUI / "roda_video.py"), str(arq), "--tomada-unica", "--saida", str(pasta),
                        "--detector", str(detector), "--contador", str(contador), "--nado", a.nado]
        cmd += ["--nadadores", str(a.nadadores)] if a.nadadores else []
        cmd += ["--sem-video"] if a.sem_video else []
        if subprocess.run(cmd).returncode != 0:
            linhas.append(dict(linha, situacao="falhou (veja a mensagem acima)", pasta=str(pasta)))
            continue
        r = json.loads((pasta / "resumo.json").read_text(encoding="utf-8"))
        linhas.append(dict(linha, situacao="contada", nadadores=r["nadadores"], origem_do_numero=r["nadadores_origem"], pasta=str(pasta)))
        for l in r["raias"]:
            bracadas.append(dict({"tomada": k, "inicio_da_tomada_s": linha["inicio_s"]},
                                 **{c: v for c, v in l.items() if c not in ("video", "instantes_s")}))
            instantes += [[k, l["raia"], l["usou"], j + 1, f"{t:.3f}", f"{t + q0 / fps:.3f}"] for j, t in enumerate(l["instantes_s"])]
    campos = ["tomada", "inicio_s", "fim_s", "quadros", "situacao", "nadadores", "origem_do_numero", "pasta"]
    with open(saida / "tomadas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    if bracadas:
        with open(saida / "bracadas_por_tomada.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bracadas[0]))
            w.writeheader()
            w.writerows(bracadas)
        with open(saida / "instantes_por_tomada.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tomada", "raia", "contador", "numero", "tempo_na_tomada_s", "tempo_no_video_s"])
            w.writerows(instantes)
    aviso = ("a numeração das raias vale dentro de cada tomada: o sistema não sabe se a raia 3 de uma tomada é o mesmo "
             "nadador da raia 3 de outra")
    (saida / "resumo.json").write_text(json.dumps({
        "feito_por": "roda_video.py (sistema final do TCC)", "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": str(video), "cortes_quadros": cortes, "cortes_s": [round(q / fps, 3) for q in cortes],
        "tomadas": linhas, "avisos": [aviso]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n===== tomadas =====")
    for l in linhas:
        extra = f" | {l['nadadores']} nadadores ({l['origem_do_numero']})" if "nadadores" in l else ""
        print(f"tomada {l['tomada']}: {l['inicio_s']:.2f} a {l['fim_s']:.2f} s | {l['situacao']}{extra}")
    print("AVISO: " + aviso)
    print(f"resultados em: {saida} (tomadas.csv, bracadas_por_tomada.csv e uma pasta por tomada)")
    return 0


def roda_detector(video, n_frames, arq_csv, arq_json, detector, chave, refaz, R, cv2):
    """Grava deteccoes.csv (formato do notebook: frame,x1,y1,x2,y2,conf) e deteccoes.json. Devolve o registro, com
    "refeita_nesta_execucao". A detecção guardada só é reaproveitada se for do mesmo vídeo, dos mesmos pesos, do mesmo
    tamanho de imagem e do mesmo rastreamento.py, e se o CSV não tiver mudado desde que foi gravado."""
    if not refaz and arq_csv.is_file() and arq_json.is_file():
        reg = json.loads(arq_json.read_text(encoding="utf-8"))
        if all(reg.get(k) == v for k, v in chave.items()) and reg.get("csv_sha1") == sha1(arq_csv):
            return dict(reg, refeita_nesta_execucao=False)
    from ultralytics import YOLO
    t0 = time.perf_counter()
    modelo = YOLO(str(detector))
    t_carga = time.perf_counter() - t0
    parcial = arq_csv.with_name(arq_csv.name + ".parcial")      # só vira deteccoes.csv quando termina
    cap = cv2.VideoCapture(str(video))
    lidos = caixas_total = 0
    t0 = time.perf_counter()
    with open(parcial, "w", newline="", encoding="utf-8") as arq:
        w = csv.writer(arq)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "conf"])
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            caixas, conf = R.detecta(modelo, frame)
            for (x1, y1, x2, y2), c in zip(caixas, conf):
                w.writerow([i, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{c:.4f}"])
            lidos += 1
            caixas_total += len(conf)
            if lidos % 50 == 0 or lidos == n_frames:
                print(f"  detector: {lidos} de {n_frames} quadros | {(time.perf_counter() - t0) / lidos:.2f} s por quadro", flush=True)
    cap.release()
    t_det = time.perf_counter() - t0
    parcial.replace(arq_csv)
    reg = dict(chave, video=video.name, detector=detector.name, confianca_minima=inspect.signature(R.detecta).parameters["conf"].default,
               quadros_lidos=lidos, caixas=caixas_total, csv_sha1=sha1(arq_csv), dispositivo=str(R.DEVICE),
               carregar_detector_s=round(t_carga, 1), deteccao_s=round(t_det, 1), s_por_quadro=round(t_det / max(lidos, 1), 3),
               quando=time.strftime("%Y-%m-%d %H:%M:%S"))
    arq_json.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    return dict(reg, refeita_nesta_execucao=True)


def rastreia(arq_csv, n, n_frames, R):
    """(lista de N raias {quadro: caixa}, eixo das raias, identidades do ByteTrack, % de quadros com N caixas antes das
    raias). É o laço da seção 7 do notebook."""
    por_frame = R.carrega_cache(arq_csv)
    novo = R.bytetrack(R.suprime_contidas(por_frame), n_frames, R.DET_CONF, R.TRACK_ACTIVATION, R.MATCH_THRESHOLD, R.LOST_BUFFER)
    if n == 1:
        unica = R.trilha_unica(novo)
        raias, eixo = [R.interpola(unica, R.LACUNA_MAX) if unica else {}], "-"
    else:
        import agrupa_raias
        raias, escolha = agrupa_raias.uma_por_raia_auto(novo, n, R.interpola, R.LACUNA_MAX)
        eixo = escolha["eixo"]
    return raias, eixo, len(novo), R.exato(novo.values(), n_frames, n)


def grava_video(video, destino, raias, tempos, info, R, cv2):
    """O vídeo anotado da seção 8c do notebook: caixa, número da raia e contagem correndo."""
    cap = cv2.VideoCapture(str(video))
    out = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["w"], info["h"]))
    for fr in range(info["n_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        for i, cad in enumerate(raias):
            if fr in cad:
                x1, y1, x2, y2 = (int(v) for v in cad[fr][:4])
                feitas = sum(1 for t in tempos[i] if t * info["fps"] <= fr)
                cor = R.CORES[i % 8]
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame, f"{i + 1}: {feitas}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()


def main():
    a = argumentos()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")     # saída redirecionada em página de código antiga não derruba o comando
    video, detector, contador = Path(a.video).resolve(), Path(a.detector).resolve(), Path(a.contador).resolve()
    if not video.is_file():
        para(f"não achei o vídeo: {video}")
    if not detector.is_file():
        para(f"não achei os pesos do detector: {detector}\n      Copie o arquivo para lá ou passe --detector CAMINHO.")
    if a.nadadores is not None and a.nadadores < 1:
        para("--nadadores tem de ser 1 ou mais.")
    if not a.so_deteccao:
        if a.nado is None:
            para("faltou dizer --nado crawl|borboleta (o nado é informado, não é reconhecido).")
        if not contador.is_file():
            para(f"não achei os pesos da rede de contagem: {contador}")
    saida = Path(a.saida).resolve() if a.saida else AQUI / "saidas" / video.stem
    saida.mkdir(parents=True, exist_ok=True)

    inicio = time.perf_counter()
    import cv2
    import numpy as np
    import torch
    import rastreamento as R

    avisos, tempos_s = [], {}

    def avisa(texto):                # o aviso sai na hora e de novo no fim, e fica no resumo.json
        avisos.append(texto)
        print("AVISO: " + texto, flush=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        para(f"o OpenCV não abriu o vídeo: {video}")
    info = {"fps": cap.get(cv2.CAP_PROP_FPS), "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    if info["n_frames"] <= 0 or info["fps"] <= 0:
        para(f"o OpenCV não soube dizer o número de quadros ou a taxa do vídeo ({info}); converta o vídeo para .mp4.")
    dispositivo = torch.cuda.get_device_name(0) if R.DEVICE != "cpu" else "cpu"
    print(f"vídeo: {video.name} | {info['n_frames']} quadros, {info['fps']:.2f} q/s, {info['w']}x{info['h']} | dispositivo: {dispositivo}", flush=True)

    # os pesos são conferidos antes de qualquer etapa demorada
    sha_det = sha1(detector)
    if sha_det in DETECTORES_DA_MEDIDA:
        print(f"pesos do detector: {DETECTORES_DA_MEDIDA[sha_det]}", flush=True)
    else:
        avisa(f"os pesos do detector não são os da medida (SHA-1 {sha_det}); os números do trabalho não valem para eles")
    sha_cont = None
    if not a.so_deteccao:
        sha_cont = sha1(contador)
        if sha_cont == CONTADOR_DA_MEDIDA:
            print("pesos do contador: os da medida (contador_rede_pesos.pt)", flush=True)
        else:
            avisa(f"os pesos da rede de contagem não são os da medida (SHA-1 {sha_cont})")

    # 1. detector
    chave ={"video_sha1": sha1(video), "video_bytes": video.stat().st_size, "detector_sha1": sha_det, "imgsz": R.IMGSZ, "quadros": info["n_frames"],
             "rastreamento_sha1": sha1(AQUI / "rastreamento.py")}        # a função detecta está nesse arquivo
    arq_csv = saida / "deteccoes.csv"
    det = roda_detector(video, info["n_frames"], arq_csv, saida / "deteccoes.json", detector, chave, a.refaz_deteccao, R, cv2)
    if det["quadros_lidos"] < info["n_frames"]:
        avisa(f"o vídeo diz ter {info['n_frames']} quadros, mas só {det['quadros_lidos']} foram lidos")
    if det["refeita_nesta_execucao"]:
        tempos_s["carregar_detector"], tempos_s["detector"] = det["carregar_detector_s"], det["deteccao_s"]
        print(f"detector: {det['caixas']} caixas em {det['quadros_lidos']} quadros | {det['s_por_quadro']:.2f} s por quadro", flush=True)
    else:
        print(f"detector: detecção guardada reaproveitada ({det['caixas']} caixas; feita em {det['quando']}, {det['deteccao_s']} s em {det['dispositivo']})", flush=True)
    if a.so_deteccao:
        print(f"só a detecção foi pedida. Guardada em {arq_csv}")
        return 0

    # troca de câmera: com mais de um nadador, cada tomada é contada separada; com um só, não há quem confundir
    import tomadas
    estimado = tomadas.estima_nadadores(arq_csv, info["n_frames"])
    cortes = []
    if not a.tomada_unica:
        cortes = cortes_do_video(video, chave["video_sha1"], info, saida, cv2)
        if cortes and (a.nadadores if a.nadadores is not None else estimado) != 1:
            return roda_por_tomada(a, video, cortes, info, saida, detector, contador, arq_csv, det, chave, cv2)
        if cortes:
            fps = info["fps"]
            print("troca de câmera em " + ", ".join(f"{q / fps:.2f} s" for q in cortes) + "; com um nadador só, "
                  "o vídeo é contado inteiro", flush=True)
        else:
            print("troca de câmera: nenhuma", flush=True)

    # número de nadadores: o informado ou, sem ele, o estimado pelas caixas
    if a.nadadores is None:
        if estimado < 1:
            para("o detector não achou nadadores na maior parte dos quadros; se houver, informe --nadadores N.")
        origem = "estimado"
        print(f"nadadores: {estimado}, estimado (caixas por quadro com confiança de pelo menos {tomadas.CONF_NADADORES}). "
              "Confira no vídeo; se estiver errado, rode de novo com --nadadores N (a detecção é reaproveitada).", flush=True)
    else:
        origem = "informado"
        if estimado != a.nadadores:
            avisa(f"foram informados {a.nadadores} nadadores, mas o detector acha {estimado} na maior parte dos quadros; confira")

    # 2 e 3. rastreamento, um nadador por raia
    n, n_frames = (a.nadadores if a.nadadores is not None else estimado), info["n_frames"]
    t0 = time.perf_counter()
    raias, eixo, ids, exato_antes = rastreia(arq_csv, n, n_frames, R)
    trilhas = {}
    for i, cad in enumerate(raias):                      # como na seção 8b do notebook: matriz [quadros, 4], NaN sem caixa
        caixas = np.full((n_frames, 4), np.nan)
        for fr, b in cad.items():
            if fr < n_frames:
                caixas[fr] = b[:4]
        trilhas[i] = caixas
    tempos_s["rastreamento"] = round(time.perf_counter() - t0, 1)
    com_trilha, exato = sum(1 for r in raias if r), R.exato(raias, n_frames, n)
    print(f"rastreamento: {ids} identidades do ByteTrack -> {com_trilha} de {n} raias com trilha | "
          f"quadros com os {n}: {exato}% | eixo das raias: {eixo}", flush=True)
    sem = [i + 1 for i, r in enumerate(raias) if not r]
    if sem:
        avisa(f"raia(s) sem trilha: {sem}; a contagem delas sai 0 e não vale")

    # 4. os dois contadores
    import contador_rede
    import contador_respingo
    t0 = time.perf_counter()
    res_resp = contador_respingo.conta_video_media(video, trilhas, a.nado, fps=info["fps"])
    tempos_s["respingo"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    rede = contador_rede.carrega_rede(contador)
    if R.PORTA_NADO and rede.cab_nado is None:
        para('PORTA_NADO está definida, mas estes pesos não têm a saída "está nadando".')
    res_rede = contador_rede.conta_video_rede(video, trilhas, a.nado, rede, fps=info["fps"], ciclos_max=R.CICLOS_MAX, porta=R.PORTA_NADO)
    tempos_s["rede"] = round(time.perf_counter() - t0, 1)

    # 5. regra final, raia a raia (seção 8c do notebook)
    linhas, tempos_finais = [], {}
    for i in range(len(raias)):
        irregular = res_rede[i]["cv_ritmo"]
        vale_rede = irregular <= R.CORTE_RITMO
        total = contador_rede.cascata(res_rede[i]["bracadas"], irregular, res_resp[i]["bracadas"], R.CORTE_RITMO)
        tempos_finais[i] = (res_rede if vale_rede else res_resp)[i]["tempos_s"]
        tem = ~np.isnan(trilhas[i][:, 0])                # onde o nadador está na imagem: centro mediano da caixa
        cx = round(float(np.median((trilhas[i][tem, 0] + trilhas[i][tem, 2]) / 2))) if tem.any() else None
        cy = round(float(np.median((trilhas[i][tem, 1] + trilhas[i][tem, 3]) / 2))) if tem.any() else None
        linhas.append({"video": video.name, "raia": i + 1, "nado": a.nado, "cobertura_pct": round(100 * len(raias[i]) / n_frames, 1),
                       "rede": res_rede[i]["bracadas"],
                       "irregularidade_do_ritmo": round(irregular, 3) if np.isfinite(irregular) else None,
                       "respingo": round(res_resp[i]["bracadas"], 1), "usou": "rede" if vale_rede else "respingo",
                       "bracadas": round(total, 1), "centro_x": cx, "centro_y": cy})

    # saídas
    pasta = saida
    with open(pasta / "bracadas.csv", "w", newline="", encoding="utf-8") as arq:
        w = csv.DictWriter(arq, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    with open(pasta / "instantes.csv", "w", newline="", encoding="utf-8") as arq:
        w = csv.writer(arq)
        w.writerow(["raia", "contador", "numero", "tempo_s"])
        for i, l in enumerate(linhas):
            for k, t in enumerate(tempos_finais[i]):
                w.writerow([i + 1, l["usou"], k + 1, f"{t:.3f}"])
    with open(pasta / "trilhas.csv", "w", newline="", encoding="utf-8") as arq:
        w = csv.writer(arq)
        w.writerow(["quadro", "raia", "x1", "y1", "x2", "y2"])
        for fr in range(n_frames):
            for i in range(len(raias)):
                if not np.isnan(trilhas[i][fr, 0]):
                    w.writerow([fr, i + 1] + [f"{v:.1f}" for v in trilhas[i][fr]])
    np.savez_compressed(pasta / "trilhas.npz", trilhas=np.stack([trilhas[i] for i in range(len(raias))]))
    anotado = None
    if not a.sem_video:
        t0 = time.perf_counter()
        anotado = pasta / f"{video.stem}_anotado.mp4"
        grava_video(video, anotado, raias, tempos_finais, info, R, cv2)
        tempos_s["video_anotado"] = round(time.perf_counter() - t0, 1)
    tempos_s["total"] = round(time.perf_counter() - inicio, 1)

    resumo = {
        "feito_por": "roda_video.py (sistema final do TCC)", "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
        "codigo_sha1": {p.name: sha1(p) for p in sorted(AQUI.glob("*.py"))},      # com que arquivos este resultado foi feito
        "video": {"arquivo": str(video), "sha1": chave["video_sha1"], "bytes": chave["video_bytes"], "quadros": n_frames,
                  "fps": info["fps"], "largura": info["w"], "altura": info["h"], "duracao_s": round(n_frames / info["fps"], 2)},
        "nadadores": n, "nadadores_origem": origem, "nadadores_estimado": estimado, "nado": a.nado,
        "trocas_de_camera_s": [round(q / info["fps"], 3) for q in cortes],
        "parametros": {k: getattr(R, k) for k in ("IMGSZ", "DET_CONF", "TRACK_ACTIVATION", "MATCH_THRESHOLD", "LOST_BUFFER", "LACUNA_MAX",
                                                  "CICLOS_MAX", "CORTE_RITMO", "PORTA_NADO")},
        "pesos": {"detector": {"arquivo": str(detector), "sha1": sha_det, "e_o_da_medida": sha_det in DETECTORES_DA_MEDIDA},
                  "contador": {"arquivo": str(contador), "sha1": sha_cont, "e_o_da_medida": sha_cont == CONTADOR_DA_MEDIDA}},
        "ambiente": {"python": platform.python_version(), "numpy": im.version("numpy"), "opencv": cv2.__version__, "scipy": im.version("scipy"),
                     "torch": torch.__version__, "ultralytics": im.version("ultralytics"), "supervision": im.version("supervision"),
                     "dispositivo": dispositivo},
        "deteccao": det,
        "rastreamento": {"identidades_do_bytetrack": ids, "quadros_com_todos_pct_antes_das_raias": exato_antes,
                         "raias_com_trilha": com_trilha, "quadros_com_todos_pct": exato, "eixo_das_raias": eixo},
        "raias": [dict(l, instantes_s=[round(t, 3) for t in tempos_finais[i]]) for i, l in enumerate(linhas)],
        "tempos_s": tempos_s, "avisos": avisos,
        "arquivos": {"bracadas": str(pasta / "bracadas.csv"), "instantes": str(pasta / "instantes.csv"), "trilhas": str(pasta / "trilhas.csv"),
                     "video_anotado": str(anotado) if anotado else None},
    }
    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"contagem ({a.nado}): regra final = rede se a irregularidade do ritmo for até {R.CORTE_RITMO}; senão, respingo")
    print(f"{'raia':>4} {'cobertura %':>11} {'rede':>6} {'irregularidade':>15} {'respingo':>9} {'valeu':>9} {'braçadas':>9}   centro (x, y)")
    for l in linhas:
        irr = "-" if l["irregularidade_do_ritmo"] is None else f"{l['irregularidade_do_ritmo']:.3f}"
        onde = "-" if l["centro_x"] is None else f"({l['centro_x']}, {l['centro_y']})"
        print(f"{l['raia']:>4} {l['cobertura_pct']:>11.1f} {l['rede']:>6.0f} {irr:>15} {l['respingo']:>9.1f} {l['usou']:>9} {l['bracadas']:>9.1f}   {onde}")
    print("tempos: " + " | ".join(f"{k.replace('_', ' ')} {v} s" for k, v in tempos_s.items()))
    for av in avisos:
        print("AVISO (de novo): " + av)
    print(f"resultados em: {pasta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
