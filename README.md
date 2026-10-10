# Pipeline de Tracking de Nadadores — TCC

Pipeline completo (pós-detecção) para o TCC de visão computacional: **tracking**,
**pose estimation / contagem de braçadas** e **avaliação**, construído em cima do
detector YOLO11 já treinado no TCC1.

## Status atual: fase HIPOTÉTICA

Todo o código abaixo já roda de ponta a ponta hoje, mas usando **dados sintéticos**
(vídeo + ground-truth gerados artificialmente) em vez do vídeo real e dos pesos
treinados de verdade. Isso permite validar a arquitetura, os testes e as métricas
antes de existir hardware/servidor para rodar o modelo real.

```
Detecção (YOLO11, já treinado no TCC1)
        │
        ▼
Tracking (ByteTrack + fallback de re-ID por aparência)   <- este pipeline
        │
        ▼
Pose por track (YOLO11n-pose, recorte por bbox)          <- este pipeline
        │
        ▼
Contagem de braçadas (oscilação dos punhos)               <- este pipeline
        │
        ▼
Avaliação (mAP, MOTA, IDF1, HOTA) + vídeo anotado          <- este pipeline
```

## Estrutura

```
configs/pipeline.yaml         # config única — muda tudo daqui, sem tocar código
src/
  detection/detector.py       # YOLO11 real + detector sintético (mesma interface)
  tracking/tracker.py         # ByteTrack + fallback de re-ID por histograma HSV
  pose/pose_estimator.py      # YOLO11n-pose real + versão sintética (a partir do GT)
  pose/stroke_counter.py      # contagem de braçadas por detecção de picos
  evaluation/metrics.py       # mAP, MOTA, IDF1, HOTA (implementações próprias/leves)
  pipeline/run_pipeline.py    # orquestra tudo
  utils/                      # config, vídeo, visualização
scripts/
  generate_synthetic_data.py  # gera o vídeo + GT sintéticos (fase hipotética)
  generate_report_plots.py    # gera gráficos (mAP, MOTA/IDF1/HOTA, braçadas) para o TCC
  run_demo.py                 # ponto de entrada único (CLI)
tests/
  test_pipeline.py            # 11 testes: componentes + integração completa (modo sintético)
  test_real_mode_integration.py  # 3 testes: dry-run do modo "real" com pesos genéricos
```

## Como rodar agora (modo sintético)

```bash
pip install -r requirements.txt
python scripts/run_demo.py --regenerate-synthetic
python scripts/generate_report_plots.py     # gera outputs/plots/*.png
pytest -v                                    # todos os testes
pytest -v -m "not slow"                      # só os rápidos (sem baixar pesos extras)
```

Saídas em `outputs/`: `annotated_output.mp4`, `results.json`, `run_config.json`, `predictions_mot.txt`, `tracks.csv`, `stroke_counts.csv`, `stroke_events.csv`,
`plots/metrics_bar.png`, `plots/ap_curve.png`, `plots/stroke_counts.png`.

Antes de rodar um vídeo real, valide os caminhos e parâmetros sem iniciar a inferência:

```bash
venv/bin/python scripts/run_demo.py \
  --config configs/pipeline_real_bootstrap.yaml --validate-only
```

O arquivo `run_config.json` registra a configuração usada em cada execução. O bootstrap
com `yolo11n.pt` e `yolo11n-pose.pt` valida a integração, mas não representa a avaliação
final do TCC; depois, basta substituir os caminhos pelos pesos treinados.

O detector real mantém a interface `detect(frame, frame_id)`, usada pelo ByteTrack, e
também oferece `detect_with_metadata(...)`. A segunda forma retorna caixas em `xyxy`,
`cxcywh`, formato normalizado, scores, classes e tempos de pré-processamento, inferência
e pós-processamento. Também existem iteradores para vídeos e sequências de imagens:
`detect_video(...)` e `detect_images(...)`.

### Ambiente CPU e PyTorch

Para este computador, use [requirements-cpu.txt](requirements-cpu.txt). O problema
observado era a combinação de `torch 2.14.0+cpu` com o wheel CUDA de `torchvision 0.29.0`;
o operador `torchvision::nms` não era carregado. Para recriar o ambiente CPU:

```bash
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements-cpu.txt
```

Para corrigir o ambiente atual sem recriá-lo:

```bash
venv/bin/pip install --force-reinstall --no-deps \
  torchvision==0.29.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

Verifique o operador e a integração com:

```bash
venv/bin/python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(hasattr(torch.ops.torchvision, 'nms'))"
venv/bin/python -m pytest -q
```

Para avaliar braçadas, crie `data/annotations/stroke_events_gt.csv` com o cabeçalho
`track_id,frame_number`, usando a numeração de frames iniciada em 1. O pipeline compara
esses eventos com tolerância configurável (`evaluation.stroke_tolerance_frames`) e salva
MAE, precisão, recall e F1 no `results.json`.

As regiões de interesse e raias usam coordenadas normalizadas entre 0 e 1. A ROI global
fica em `roi.points`; para restringir a piscina por raia, preencha `roi.lanes` com um
polígono por raia na configuração real. A filtragem geométrica já está disponível, mas a
validação final da identidade entre raias depende de um trecho real anotado.

Para calibrar raias em outro vídeo, use o primeiro frame estável da piscina:

```bash
venv/bin/python scripts/calibrate_lanes.py \
  --video data/raw/video_natacao_trecho.mp4 \
  --output data/annotations/lanes.json --lanes 8
```

Copie o campo `lanes` gerado para `roi.lanes`. Com `tracker.lane_constraint_enabled: true`,
um ID não atravessa uma raia configurada; se o detector perder o atleta e gerar outro ID
na mesma raia, o pipeline tenta recuperar a identidade por proximidade e aparência.
Os campos `head_confidence`, `shoulder_confidence` e `arm_confidence` em `tracks.csv`
permitem auditar a qualidade da pose e detectar quando cabeça ou braços estão invisíveis.

Para executar todos os cortes `*_c*.mp4` sem aplicar a ROI específica de um único vídeo:

```bash
venv/bin/python scripts/run_batch.py \
  --config configs/pipeline_real_video_natacao.yaml \
  --input-dir data/raw \
  --output-dir outputs_batch \
  --roi-mode full
```

O lote gera uma pasta por vídeo e os arquivos `batch_summary.csv` e
`batch_summary.json`. Use `--roi-mode config` somente quando a geometria da configuração
for válida para todos os vídeos selecionados. Use `--save-video` para guardar os vídeos
anotados, ou omita essa opção para priorizar métricas e economizar disco.

Para testar a calibração automática de raias em todos os cortes, acrescente:

```bash
--lane-mode auto
```

O padrão `--lane-mode config` preserva o comportamento da configuração. Use
`--lane-mode disabled` para garantir que nenhum filtro de raia seja aplicado.
Os metadados da execução informam `detected_polygons`, `line_count`, `water_ratio`
e `confidence` da calibração.

O campo `persistent_reassociations` mede quantas vezes um novo ID interno do ByteTrack
foi absorvido por uma identidade lógica já existente. `new_logical_identities` é o número
de identidades expostas ao pipeline; ele deve ser interpretado junto com a contagem real
de nadadores, não como uma medida de acurácia isolada.

Em CPU, `pose.inference_interval_frames: 3` é uma opção prática para executar pose a
cada três frames e preservar a detecção/tracking em todos os frames. Para avaliação final
de braçadas, use intervalo 1 ou valide que o intervalo escolhido não perde eventos.

## Decisões de projeto (por que está assim)

- **Tudo leve, de propósito.** Tracking por movimento/IoU (ByteTrack) em vez de
  re-ID por rede neural pesada; re-ID de fallback via histograma de cor (não
  embedding de deep learning); pose com YOLO11n-pose (6MB) em vez de modelos
  maiores. A ideia é rodar em CPU sem servidor dedicado nesta fase; a evolução
  para modelos maiores fica para o TCC2, quando houver servidor disponível.
- **Fallback de re-ID por aparência** (`src/tracking/tracker.py`): implementa a
  ideia documentada no TCC1 ("fallback framing for embedding-based association")
  — o ByteTrack resolve a maioria dos casos por movimento; só quando ele *não*
  consegue casar uma detecção nova a nenhuma track ativa é que entra a
  comparação de aparência contra tracks perdidas recentemente.
- **Modelo de pose ainda não escolhido de verdade** (decisão documentada como
  adiada para o TCC2). Por isso existe `SyntheticPoseEstimator`, que usa os
  keypoints de punho do GT sintético — o `StrokeCounter` e as métricas já ficam
  validados e prontos, e trocar para um modelo real de pose depois é só trocar
  a peça (`Yolo11PoseEstimator`, já implementado) na config.
- **HOTA implementado do zero** (não usei o TrackEval oficial): a biblioteca de
  referência é pesada e vai além do escopo "leve" do projeto. A implementação
  em `metrics.py` segue a decomposição DetA × AssA da métrica original
  (Luiten et al., 2020), simplificada para o caso single-class. Documentar essa
  simplificação no texto do TCC é importante.
- **Testes de integração do modo "real" já existem** (`tests/test_real_mode_integration.py`),
  mesmo sem os pesos treinados de verdade: eles usam o YOLO11n/YOLO11n-pose
  genéricos (pré-treinados no COCO) só para validar que o código de
  carregamento/inferência/conversão não quebra. Isso já pegou e evitou bugs de
  integração antes mesmo de existir o modelo treinado real disponível aqui.

## Migração para o caso real (quando tiver o servidor)

Em teoria, **nenhum código muda** — só a config:

1. `mode: "real"` em `configs/pipeline.yaml`
2. `detector.weights_path` → caminho do `.pt` treinado no TCC1
3. `video.source` → caminho do vídeo real
4. Se for avaliar contra ground-truth real, `evaluation.gt_path` → anotações reais
   no formato MOTChallenge (frame, id, x, y, w, h, conf, class, visibility —
   veja `data/annotations/gt_mot.txt` sintético como exemplo do formato)
5. (Quando decidir o modelo de pose real) `pose.weights_path` → pesos do modelo
   escolhido; se não for YOLO11-pose, criar uma nova classe implementando
   `BasePoseEstimator` em `pose_estimator.py` (mesma interface)

Pontos de atenção nessa migração (vale já deixar anotado no texto do TCC):

- **Threshold do stroke counter** (`min_peak_prominence`, `min_peak_distance_sec`)
  foi calibrado no sinal sintético (limpo); com keypoints reais (mais ruidosos),
  provavelmente vai precisar recalibrar ou aumentar `smoothing_window`.
- **`sv.ByteTrack` está deprecated** no `supervision` (remoção prevista pra
  v0.31.0, sem substituto direto anunciado — ver comentário em `tracker.py`).
  Fixamos a versão em `requirements.txt`; revisar antes de atualizar o pacote.
- **Múltiplas pessoas por crop na pose real**: se duas raias ficarem muito
  próximas, o crop de uma track pode capturar parte de outro nadador; hoje o
  `Yolo11PoseEstimator` pega a maior detecção de pessoa dentro do crop como
  heurística — validar se isso é suficiente com vídeo real.
