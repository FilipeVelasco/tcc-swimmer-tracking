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

Saídas em `outputs/`: `annotated_output.mp4`, `results.json`, `stroke_counts.csv`,
`plots/metrics_bar.png`, `plots/ap_curve.png`, `plots/stroke_counts.png`.

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
