import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

try:
    import psutil
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    ERRO_IMPORT = None
except ImportError as erro:  # a mensagem sai no começo de main()
    ERRO_IMPORT = erro

# ---------- receita: a mesma do treino final ----------
PESOS_INICIAIS = 'yolo11m.pt'
IMGSZ = 640
EPOCAS = 25
BATCH = -1                     # -1 = maior lote que cabe na GPU
WORKERS = 2
FRACAO = 1.0
COPIAS = 9                     # no ZIP cada quadro do treino tem 9 cópias aumentadas; os da validação e do teste têm 1
VERSAO_DO_TREINO_FINAL = '8.4.211'

# Ensaio rápido em CPU, só para testar o script (não é medida): TCC_SMOKE=1
ENSAIO = os.environ.get('TCC_SMOKE') == '1'
if ENSAIO:
    EPOCAS = int(os.environ.get('TCC_SMOKE_EPOCAS', '1'))
    IMGSZ, BATCH, FRACAO = 160, 4, 0.005

# ---------- onde estão os arquivos ----------
BASE = Path(os.environ.get('TCC_BASE', Path.home() / 'TCC-nadador'))
ZIP_PATH = BASE / 'Dataset' / 'dataset nadador Yolo11.zip'
TRABALHO = BASE / 'trabalho' / 'validacao_cruzada'
TRAVA = BASE / 'trabalho' / 'validacao_cruzada_em_execucao.json'
SAIDA = BASE / 'resultados' / 'validacao_cruzada_detector'
RESUMO = BASE / 'resultados' / 'validacao_cruzada_detector_resumo'

# ---------- o conjunto e as dobras (fixados antes de rodar) ----------
SHA1_ZIP = '89cb637042fc68996487151f72ae02dc919a944f'
QUADROS_POR_NOME = {
    'varie-te-trop-zoome-': 68, 'yt-QPU-BNrkFeI': 55, 'StRaphael18': 45, 'swimmingGameLastPart': 44,
    'Rennes19': 42, 'Montpellier18': 40, 'Angers19': 38, '7Collines19': 36, 'Stanford19': 35,
    'Stanforf19': 35, 'Gwangju': 14, 'DJI': 7, 'Nice': 6, 'GOLDEN-TOUR': 2,
}
GRUPO_DO_NOME = {'GOLDEN-TOUR': 'Nice', 'Stanforf19': 'Stanford19'}
SEMPRE_NO_TREINO = {'DJI'}
DOBRAS = {
    1: ('Angers19', 'Gwangju', 'Nice'),
    2: ('Stanford19', '7Collines19'),
    3: ('varie-te-trop-zoome-', 'Montpellier18'),
    4: ('yt-QPU-BNrkFeI', 'Rennes19'),
    5: ('StRaphael18', 'swimmingGameLastPart'),
}
QUADROS_NO_TESTE = {1: 60, 2: 106, 3: 108, 4: 97, 5: 89}
NOMES_DO_TESTE_ORIGINAL = ('Angers19', 'Gwangju', 'Nice')   # sem GOLDEN-TOUR: as 58 imagens do teste original
EXT_IMAGEM = {'.jpg', '.jpeg', '.png'}


def origem(p):
    """Quadro de origem: nome do arquivo sem o hash do Roboflow."""
    return re.sub(r'\.rf\.[a-zA-Z0-9]+$', '', Path(p).stem)


def quadro(arquivo):
    """Quadro de origem de uma imagem do conjunto montado (sem o sufixo _copiaN das repetições)."""
    return origem(re.sub(r'_copia\d+$', '', Path(arquivo).stem) + Path(arquivo).suffix)


def competicao(p):
    base = re.sub(r'_(jpg|jpeg|png)$', '', origem(p))
    prova = re.sub(r'[-_]\d+$', '', base)
    return prova.split('_')[0]


def grupo(nome):
    return GRUPO_DO_NOME.get(nome, nome)


def imagens(pasta):
    return sorted(p for p in Path(pasta).glob('*.*') if p.suffix.lower() in EXT_IMAGEM)


def agora():
    return time.strftime('%d/%m/%Y %H:%M:%S')


def sha1(caminho):
    h = hashlib.sha1()
    with open(caminho, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def grava_json(caminho, obj):
    tmp = caminho.with_name(caminho.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, caminho)


def log(msg):
    linha = f'[{agora()}] {msg}'
    print(linha, flush=True)
    RESUMO.mkdir(parents=True, exist_ok=True)
    with open(RESUMO / 'progresso.txt', 'a', encoding='utf-8') as f:
        f.write(linha + '\n')


def para(msg):
    log('ERRO: ' + msg)
    sys.exit(1)


def num(x, casas=3):
    return f'{x:.{casas}f}'.replace('.', ',')


# ---------- trava: impede duas execuções ao mesmo tempo ----------
def pega_trava():
    if TRAVA.exists():
        try:
            t = json.loads(TRAVA.read_text(encoding='utf-8'))
            p = psutil.Process(int(t['pid']))
            if abs(p.create_time() - float(t['inicio_processo'])) < 1:
                para(f"o script já está rodando em outra janela (processo {t['pid']}, desde {t['desde']}). "
                     'Feche esta janela e deixe a outra terminar.')
        except (psutil.Error, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        log('AVISO: a execução anterior parou no meio; esta continua de onde ela parou.')
    TRAVA.parent.mkdir(parents=True, exist_ok=True)
    eu = psutil.Process()
    grava_json(TRAVA, {'pid': eu.pid, 'inicio_processo': eu.create_time(), 'desde': agora()})


def solta_trava():
    try:
        if TRAVA.exists() and json.loads(TRAVA.read_text(encoding='utf-8')).get('pid') == os.getpid():
            TRAVA.unlink()
    except (OSError, ValueError):
        pass


# ---------- 1. conferência do ambiente (primeiro minuto) ----------
def confere_ambiente():
    log('Validação cruzada do detector por competição' + (' (ENSAIO em CPU, não é medida)' if ENSAIO else ''))
    log(f'Pasta: {BASE}')
    log(f'Python {sys.version.split()[0]} ({sys.executable})')
    log(f'torch {torch.__version__} | ultralytics {ultralytics.__version__}')
    if ultralytics.__version__ != VERSAO_DO_TREINO_FINAL:
        log(f'AVISO: o modelo final foi treinado com o ultralytics {VERSAO_DO_TREINO_FINAL}; esta versão é outra. '
            'O script continua, e a versão fica registrada.')
    if torch.cuda.is_available():
        device = 0
        log(f'OK  GPU: {torch.cuda.get_device_name(0)}')
    elif ENSAIO:
        device = 'cpu'
        log('AVISO: sem GPU; ensaio em CPU')
    else:
        para('o PyTorch não está vendo a GPU, e sem ela o treino levaria dias. Confira com: '
             'python -c "import torch; print(torch.cuda.is_available())"')
    if not ZIP_PATH.exists():
        para(f'não achei o conjunto de dados em {ZIP_PATH}')
    sha_zip = sha1(ZIP_PATH)
    if sha_zip == SHA1_ZIP:
        log(f'OK  conjunto de dados: {ZIP_PATH.name}')
    else:
        log(f'AVISO: o {ZIP_PATH.name} tem SHA-1 {sha_zip}, diferente do esperado. A contagem de quadros '
            'por competição, logo abaixo, confere se o conteúdo é o mesmo.')
    pesos = next((p for p in (BASE / PESOS_INICIAIS, Path.cwd() / PESOS_INICIAIS) if p.exists()), None)
    if pesos:
        log(f'OK  pesos iniciais: {pesos}')
    else:
        log(f'AVISO: não achei {PESOS_INICIAIS} em {BASE}; o Ultralytics vai baixá-lo (40 MB, precisa de internet).')
    livre = shutil.disk_usage(BASE).free / 1e9
    if livre < 10:
        para(f'há só {livre:.1f} GB livres no disco; são precisos uns 10 GB (o cache das imagens ocupa cerca de 6 GB).')
    log(f'OK  disco: {livre:.0f} GB livres')
    ambiente = {
        'inicio': agora(), 'python': sys.version.split()[0], 'executavel': sys.executable,
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'sha1_zip': sha_zip, 'pesos_iniciais': str(pesos) if pesos else PESOS_INICIAIS,
        'sha1_pesos_iniciais': sha1(pesos) if pesos else None, 'sha1_script': sha1(Path(__file__)),
        'ensaio': ENSAIO, 'receita': receita(),
    }
    anteriores = []
    if (RESUMO / 'ambiente.json').exists():
        anteriores = json.loads((RESUMO / 'ambiente.json').read_text(encoding='utf-8'))
    grava_json(RESUMO / 'ambiente.json', anteriores + [ambiente])
    return device, str(pesos) if pesos else PESOS_INICIAIS


def receita():
    return {'pesos_iniciais': PESOS_INICIAIS, 'imgsz': IMGSZ, 'epocas': EPOCAS, 'batch': BATCH, 'workers': WORKERS,
            'fracao': FRACAO, 'patience': 0, 'cos_lr': True, 'close_mosaic': 10, 'cache': 'disk', 'amp': True,
            'seed': 42, 'pesos_avaliados': 'last.pt'}


# ---------- 2. preparação: todas as imagens numa pasta só, e uma lista por dobra ----------
def prepara():
    marca = TRABALHO / 'preparado.json'
    if marca.exists():
        info = json.loads(marca.read_text(encoding='utf-8'))
        log(f"OK  dados já preparados ({info['imagens_no_conjunto']} imagens de {info['quadros']} quadros)")
        return info
    log('Preparando os dados (1 a 3 minutos)...')
    if TRABALHO.exists():
        shutil.rmtree(TRABALHO)
    bruto, todas, listas = TRABALHO / 'bruto', TRABALHO / 'todas', TRABALHO / 'listas'
    bruto.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(bruto)
    yaml_orig = sorted(bruto.rglob('data.yaml'))[0]
    raiz = yaml_orig.parent
    nomes = yaml.safe_load(yaml_orig.read_text(encoding='utf-8')).get('names', [])
    # nadador = bodysurface ou bodyunder; a classe Relax fica de fora, como no treino final
    mapa = {i: 0 if n.lower() in ('bodysurface', 'bodyunder') or len(nomes) == 1 else -1 for i, n in enumerate(nomes)}
    for d in ('images', 'labels'):
        (todas / d).mkdir(parents=True)

    copias = Counter(origem(img) for s in ('train', 'valid', 'test') for img in imagens(raiz / s / 'images'))
    quadros, caixas = {}, {}
    for s in ('train', 'valid', 'test'):
        for img in imagens(raiz / s / 'images'):
            lbl = raiz / s / 'labels' / f'{img.stem}.txt'
            linhas = []
            for linha in (lbl.read_text().splitlines() if lbl.exists() else []):
                p = linha.split()
                if p and mapa.get(int(p[0]), -1) != -1:
                    linhas.append(f"0 {' '.join(p[1:])}\n")
            repeticoes = COPIAS if copias[origem(img)] == 1 else 1
            q = quadros.setdefault(origem(img), {'nome': competicao(img), 'split': s, 'imagens': []})
            for k in range(repeticoes):
                nome = img.stem if repeticoes == 1 else f'{img.stem}_copia{k + 1}'
                destino = todas / 'images' / f'{nome}{img.suffix}'
                shutil.copy(img, destino)
                (todas / 'labels' / f'{nome}.txt').write_text(''.join(linhas))
                q['imagens'].append(str(destino))
                caixas[str(destino)] = len(linhas)

    por_nome = dict(Counter(q['nome'] for q in quadros.values()))
    if por_nome != QUADROS_POR_NOME:
        para(f'o conjunto de dados não é o esperado. Quadros por competição: {por_nome}')
    n_imagens = sum(len(q['imagens']) for q in quadros.values())
    if n_imagens != len(quadros) * COPIAS:
        para(f'esperava {len(quadros) * COPIAS} imagens no conjunto montado e há {n_imagens}')

    listas.mkdir()

    def escreve_lista(nome, arquivos):
        caminho = listas / f'{nome}.txt'
        caminho.write_text('\n'.join(sorted(arquivos)) + '\n', encoding='utf-8')
        return caminho

    def escreve_yaml(nome, treino, validacao, teste):
        caminho = listas / f'{nome}.yaml'
        caminho.write_text(yaml.safe_dump({'train': str(treino), 'val': str(validacao), 'test': str(teste), 'nc': 1,
                                           'names': ['swimmer']}, sort_keys=False, allow_unicode=True), encoding='utf-8')
        return str(caminho)

    dobras = {}
    for k, grupos_teste in DOBRAS.items():
        de_fora = [q for q in quadros.values() if grupo(q['nome']) in grupos_teste]
        no_treino = [q for q in quadros.values() if grupo(q['nome']) not in grupos_teste]
        treino = [im for q in no_treino for im in q['imagens']]
        validacao = [q['imagens'][0] for q in no_treino if q['split'] != 'train']
        teste = [q['imagens'][0] for q in de_fora]
        if len(teste) != QUADROS_NO_TESTE[k] or len(treino) != (len(quadros) - QUADROS_NO_TESTE[k]) * COPIAS:
            para(f'dobra {k}: {len(teste)} imagens de teste e {len(treino)} de treino, fora do esperado')
        if {quadro(im) for im in treino} & {quadro(im) for im in teste}:
            para(f'dobra {k}: há quadro no treino e no teste')
        if Counter(Counter(quadro(im) for im in treino).values()) != Counter({COPIAS: len(no_treino)}):
            para(f'dobra {k}: algum quadro do treino não tem {COPIAS} imagens')
        if any(grupo(q['nome']) in grupos_teste for q in no_treino) or any(q['nome'] in SEMPRE_NO_TREINO for q in de_fora):
            para(f'dobra {k}: grupo no lugar errado')
        l_treino = escreve_lista(f'dobra_{k}_treino', treino)
        l_val = escreve_lista(f'dobra_{k}_validacao_dentro_do_treino', validacao)
        l_teste = escreve_lista(f'dobra_{k}_teste', teste)
        info = {
            'grupos_no_teste': list(grupos_teste),
            'nomes_no_teste': sorted({q['nome'] for q in de_fora}),
            'imagens_treino': len(treino), 'quadros_treino': len(no_treino),
            'imagens_validacao_dentro_do_treino': len(validacao),
            'imagens_teste': len(teste), 'caixas_teste': sum(caixas[im] for im in teste),
            'yaml': escreve_yaml(f'dobra_{k}', l_treino, l_val, l_teste),
            'por_grupo': {},
        }
        for g in grupos_teste:
            so_g = [q['imagens'][0] for q in de_fora if grupo(q['nome']) == g]
            l_g = escreve_lista(f'dobra_{k}_teste_{g}', so_g)
            info['por_grupo'][g] = {'imagens': len(so_g), 'caixas': sum(caixas[im] for im in so_g),
                                    'yaml': escreve_yaml(f'dobra_{k}_{g}', l_treino, l_val, l_g)}
        if k == 1:
            original = [q['imagens'][0] for q in de_fora if q['nome'] in NOMES_DO_TESTE_ORIGINAL]
            if len(original) != 58:
                para(f'o teste original deveria ter 58 imagens e tem {len(original)}')
            l_o = escreve_lista('dobra_1_teste_original_58', original)
            info['teste_original_58'] = {'imagens': len(original), 'caixas': sum(caixas[im] for im in original),
                                         'yaml': escreve_yaml('dobra_1_teste_original_58', l_treino, l_val, l_o)}
        dobras[str(k)] = info

    shutil.rmtree(bruto)
    info = {'preparado_em': agora(), 'imagens_no_conjunto': n_imagens, 'quadros': len(quadros),
            'quadros_por_nome': por_nome, 'grupo_do_nome': GRUPO_DO_NOME, 'sempre_no_treino': sorted(SEMPRE_NO_TREINO),
            'dobras': dobras}
    grava_json(RESUMO / 'dobras.json', info)
    grava_json(marca, info)
    log(f'OK  dados preparados: {n_imagens} imagens de {len(quadros)} quadros; 5 dobras conferidas')
    return info


# ---------- 3. treino de uma dobra ----------
def estado_do_treino(last):
    if not last.exists():
        return 'NOVO'
    try:
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
    except Exception:
        return 'CORROMPIDO'
    if ckpt.get('epoch', 0) == -1 or not ckpt.get('optimizer') or ckpt.get('epoch', 0) >= EPOCAS - 1:
        return 'CONCLUIDO'
    return 'RETOMAR'


def confere_receita(run):
    args = run / 'args.yaml'
    if not args.exists():
        return
    a = yaml.safe_load(args.read_text(encoding='utf-8'))
    diferente = {c: (a.get(c), v) for c, v in (('epochs', EPOCAS), ('imgsz', IMGSZ), ('fraction', FRACAO)) if a.get(c) != v}
    if diferente:
        para(f'a pasta {run} tem um treino feito com outra receita ({diferente}). '
             'Renomeie ou apague essa pasta e rode de novo.')


def treina(k, info, device, pesos):
    nome = f'dobra_{k}'
    run = SAIDA / nome
    last = run / 'weights' / 'last.pt'
    confere_receita(run)
    estado = estado_do_treino(last)
    if estado == 'CORROMPIDO':
        guardada = SAIDA / f'{nome}_interrompida_{time.strftime("%Y%m%d_%H%M%S")}'
        run.rename(guardada)
        log(f'AVISO: os pesos da {nome} estavam corrompidos; a pasta virou {guardada.name} e a dobra recomeça.')
        estado = 'NOVO'
    d = info['dobras'][str(k)]
    log(f"Dobra {k} de {len(DOBRAS)}: teste com {', '.join(d['grupos_no_teste'])} ({d['imagens_teste']} imagens); "
        f"treino com {d['imagens_treino']} imagens de {d['quadros_treino']} quadros. Estado: {estado}")
    if estado == 'RETOMAR':
        YOLO(str(last)).train(resume=True)
    elif estado == 'NOVO':
        YOLO(pesos).train(
            data=d['yaml'], epochs=EPOCAS, imgsz=IMGSZ, batch=BATCH, device=device, workers=WORKERS,
            project=str(SAIDA), name=nome, exist_ok=True, patience=0, cos_lr=True,
            close_mosaic=10, cache='disk', amp=True, plots=True, seed=42, fraction=FRACAO,
        )
    if estado_do_treino(last) != 'CONCLUIDO':
        para(f'o treino da {nome} não terminou (pesos em {last})')
    return last, estado


# ---------- 4. avaliação (mesma chamada do notebook) ----------
def avalia(pesos, yaml_teste, device):
    r = YOLO(str(pesos)).val(data=yaml_teste, split='test', imgsz=IMGSZ, device=device, plots=False,
                             verbose=False, workers=0, project=str(TRABALHO / 'val_tmp'), name='v', exist_ok=True)
    return {'precisao': float(r.box.mp), 'recall': float(r.box.mr), 'mAP50': float(r.box.map50), 'mAP50_95': float(r.box.map)}


def avalia_dobra(k, info, last, device):
    arquivo = SAIDA / f'dobra_{k}' / 'avaliacao.json'
    if arquivo.exists():
        return json.loads(arquivo.read_text(encoding='utf-8'))
    d = info['dobras'][str(k)]
    log(f'Dobra {k}: avaliando no teste ({d["imagens_teste"]} imagens)')
    res = {'dobra': k, 'grupos_no_teste': d['grupos_no_teste'], 'pesos': str(last), 'sha1_pesos': sha1(last),
           'teste': {'imagens': d['imagens_teste'], 'caixas': d['caixas_teste'], **avalia(last, d['yaml'], device)},
           'por_grupo': {}}
    for g, dg in d['por_grupo'].items():
        res['por_grupo'][g] = {'imagens': dg['imagens'], 'caixas': dg['caixas'], **avalia(last, dg['yaml'], device)}
    if 'teste_original_58' in d:
        do = d['teste_original_58']
        res['teste_original_58'] = {'imagens': do['imagens'], 'caixas': do['caixas'], **avalia(last, do['yaml'], device)}
    res['avaliado_em'] = agora()
    grava_json(arquivo, res)
    t = res['teste']
    log(f"Dobra {k}: mAP50 {num(t['mAP50'])} | mAP50-95 {num(t['mAP50_95'])} | precisão {num(t['precisao'])} | "
        f"recall {num(t['recall'])}")
    return res


def copia_para_o_resumo(k):
    run = SAIDA / f'dobra_{k}'
    for nome in ('results.csv', 'args.yaml', 'avaliacao.json'):
        if (run / nome).exists():
            shutil.copy(run / nome, RESUMO / f'dobra_{k}_{nome}')


# ---------- 5. resumo ----------
def escreve_resumo(resultados, info):
    met = ('precisao', 'recall', 'mAP50', 'mAP50_95')
    media = {m: statistics.mean(r['teste'][m] for r in resultados) for m in met}
    menor = {m: min(r['teste'][m] for r in resultados) for m in met}
    maior = {m: max(r['teste'][m] for r in resultados) for m in met}
    por_grupo = {g: {'dobra': r['dobra'], **v} for r in resultados for g, v in r['por_grupo'].items()}
    final = {'concluido_em': agora(), 'ensaio': ENSAIO, 'receita': receita(), 'dobras': resultados,
             'media_das_dobras': media, 'menor': menor, 'maior': maior, 'por_grupo': por_grupo}
    grava_json(RESUMO / 'resultado_validacao_cruzada.json', final)

    linhas = ['Validação cruzada do detector por competição' + (' (ENSAIO em CPU, não é medida)' if ENSAIO else ''),
              f"Receita: {PESOS_INICIAIS}, imgsz {IMGSZ}, {EPOCAS} épocas, sem parada antecipada, pesos da última época; "
              f"{len(DOBRAS)} dobras por grupo de competição.",
              f"Concluída em {final['concluido_em']}.", '',
              f"{'Dobra':<6}{'Competições no teste':<44}{'Imagens':>8}{'Caixas':>8}{'Precisão':>10}{'Recall':>8}"
              f"{'mAP50':>8}{'mAP50-95':>10}"]
    for r in resultados:
        t = r['teste']
        linhas.append(f"{r['dobra']:<6}{', '.join(r['grupos_no_teste']):<44}{t['imagens']:>8}{t['caixas']:>8}"
                      f"{num(t['precisao']):>10}{num(t['recall']):>8}{num(t['mAP50']):>8}{num(t['mAP50_95']):>10}")
    for rotulo, v in (('Média das dobras', media), ('Menor', menor), ('Maior', maior)):
        linhas.append(f"{rotulo:<66}{num(v['precisao']):>10}{num(v['recall']):>8}{num(v['mAP50']):>8}{num(v['mAP50_95']):>10}")
    linhas += ['', 'Por competição (cada uma medida pelo modelo que não a viu no treino):',
               f"{'Competição':<26}{'Dobra':>6}{'Imagens':>8}{'Caixas':>8}{'mAP50':>8}{'mAP50-95':>10}"]
    for g, v in sorted(por_grupo.items(), key=lambda x: x[1]['mAP50']):
        linhas.append(f"{g:<26}{v['dobra']:>6}{v['imagens']:>8}{v['caixas']:>8}{num(v['mAP50']):>8}{num(v['mAP50_95']):>10}")
    o = resultados[0].get('teste_original_58')
    if o:
        linhas += ['', f"Teste original ({o['imagens']} imagens de Angers19, Gwangju e Nice, sem GOLDEN-TOUR), "
                       f"modelo da dobra 1: mAP50 {num(o['mAP50'])}; mAP50-95 {num(o['mAP50_95'])}; "
                       f"precisão {num(o['precisao'])}; recall {num(o['recall'])}"]
    texto = '\n'.join(linhas) + '\n'
    (RESUMO / 'resultado_validacao_cruzada.txt').write_text(texto, encoding='utf-8')
    print('\n' + texto, flush=True)


# ---------- principal ----------
def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except (AttributeError, ValueError):
        pass
    if ERRO_IMPORT is not None:
        print(f'ERRO: falta um pacote ({ERRO_IMPORT}). Rode no mesmo prompt em que você abre o Jupyter.', flush=True)
        sys.exit(1)
    RESUMO.mkdir(parents=True, exist_ok=True)
    pega_trava()
    try:
        arquivo_log = logging.FileHandler(RESUMO / 'log_ultralytics.txt', encoding='utf-8')
        arquivo_log.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        LOGGER.addHandler(arquivo_log)
        device, pesos = confere_ambiente()
        info = prepara()
        resultados, duracoes = [], []
        for k in DOBRAS:
            inicio = time.time()
            last, estado = treina(k, info, device, pesos)
            resultados.append(avalia_dobra(k, info, last, device))
            copia_para_o_resumo(k)
            faltam = len(DOBRAS) - k
            if estado != 'CONCLUIDO':
                minutos = (time.time() - inicio) / 60
                if estado == 'NOVO':       # dobra retomada no meio não entra na previsão
                    duracoes.append(minutos)
                previsao = (f' Faltam {faltam}; previsão: cerca de {statistics.mean(duracoes) * faltam:.0f} min.'
                            if faltam and duracoes else '')
                log(f'Dobra {k} pronta em {minutos:.0f} min.' + previsao)
        escreve_resumo(resultados, info)
        log(f'Pronto. Suba para o Drive a pasta {RESUMO}')
    finally:
        solta_trava()


if __name__ == '__main__':
    main()
