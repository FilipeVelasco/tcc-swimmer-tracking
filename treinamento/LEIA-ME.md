# Treino dos dois modelos

O sistema usa dois modelos que nós treinamos: o detector de nadadores (YOLO11m) e a rede pequena que conta as braçadas. Esta pasta tem o código que treinou cada um. Os pesos que saíram desses treinos estão em [`../sistema_final`](../sistema_final).

Os dados não estão aqui. O conjunto de imagens do detector e a base usada no pré-treino da rede são públicos (os endereços estão abaixo). Os vídeos das provas vêm de transmissões e de bancos de imagens e também ficam de fora. O código desta pasta mostra como cada treino foi feito e permite repeti-lo, com os dados obtidos à parte.

## Detector de nadadores (`detector/`)

**Conjunto de dados.** *Swimmer-detection*, versão 1, do Roboflow Universe (`swimmingdetection/swimmer-detection-gjwyk`), sob licença CC BY 4.0: <https://universe.roboflow.com/swimmingdetection/swimmer-detection-gjwyk/dataset/1>.

- São 3.123 imagens de 640 × 640, que vêm de 467 quadros distintos de 14 nomes de competição. Na parte de treino do arquivo, cada quadro vem em 9 cópias com aumento de dados.
- Usamos a exportação em formato YOLOv11, o arquivo `dataset nadador Yolo11.zip` (SHA-1 `89cb637042fc68996487151f72ae02dc919a944f`).
- As classes `bodysurface` e `bodyunder` viram uma só, `swimmer`. A classe `Relax` fica de fora.

**Os passos.** Estão no notebook [`detector/treino_detector.ipynb`](detector/treino_detector.ipynb), que roda no Google Colab ou num PC com GPU NVIDIA. Ele espera a pasta `TCC-nadador` com o conjunto em `Dataset/dataset nadador Yolo11.zip`; a primeira célula de código diz onde.

1. **Divisão por competição** (seção 3). Nenhuma imagem de uma competição da validação ou do teste entra no treino. O teste fica com `Angers19`, `Gwangju` e `Nice`, a validação com `StRaphael18` e `7Collines19`, e o treino com as outras. São 2.264 imagens no treino, 81 na validação e 58 no teste.
2. **Treino por competição** (seções 4 a 6 e 8). O YOLO11m parte dos pesos do COCO, com imagens de 640, até 150 épocas e parada antecipada (paciência de 30 épocas), semente 42. Depois vêm a avaliação na validação e no teste e as outras arquiteturas, para comparar (YOLO11n, YOLO11s e YOLO11x).
3. **Treino final** (seção 9). É a mesma receita, agora com as 14 competições no treino (4.203 imagens), um número fixo de épocas (25 para o YOLO11m), sem parada antecipada, ficando com os pesos da última época. Esse arquivo é o `sistema_final/yolo11m_640_final_14_competicoes.pt`. Como todas as imagens entram no treino, o modelo final não tem nota de teste.

O notebook é o que treinou o detector, só com as partes de treino. As células de código são as que rodaram, sem mudança; o começo do notebook explica o que saiu. Os parâmetros do treino final ficaram gravados no próprio arquivo de pesos: YOLO11m, 25 épocas, imagens de 640, lote de 7 imagens (o maior que coube na GPU), semente 42 e otimizador AdamW, com o Ultralytics 8.4.211, em 12/11/2026.

**A nota do detector** sai de [`detector/validacao_cruzada_detector.py`](detector/validacao_cruzada_detector.py). Ele repete a receita do treino final cinco vezes, cada vez com um grupo de competições fora do treino, e mede o detector nesse grupo. Rodou uma única vez, em 16/11/2026, numa RTX 2060 SUPER: mAP@0,5 médio de 0,700 (de 0,556 a 0,747 entre as cinco rodadas).

- A divisão é por competição. Nice e GOLDEN-TOUR (Nice) contam como a mesma competição, e Stanford19 e Stanforf19 também. Os clipes de drone (DJI) ficam sempre no treino. Sobram 11 grupos, em 5 dobras fixas; a dobra 1 tem as competições do teste original (Angers19, Gwangju e Nice).
- No treino de cada dobra, cada quadro entra com 9 imagens, como no treino final. No teste, entra uma imagem por quadro das competições que ficaram de fora.
- Para rodar: `python validacao_cruzada_detector.py`, na pasta `TCC-nadador` (a mesma do notebook). Se parar no meio, o mesmo comando continua de onde parou. O resumo sai em `resultados/validacao_cruzada_detector_resumo`.

## Rede de contagem (`contador/`)

A rede recebe a sequência de recortes de um nadador (em tons de cinza, 24 × 24, com o corpo na horizontal, a 15 quadros por segundo) e devolve, quadro a quadro, a densidade de braçadas: um sinal que sobe perto de cada braçada. No treino, o alvo é uma pequena elevação centrada em cada instante marcado à mão. No sistema, cada pico desse sinal conta como uma braçada. A rede e o treino estão em [`contador/rede2.py`](contador/rede2.py).

Ela é pré-treinada com a base pública de T. Woinoski e I. V. Bajić, do artigo *Swimmer Stroke Rate Estimation From Overhead Race Video* (2021, <https://arxiv.org/abs/2104.12056>). Depois, é ajustada com essa base e os nossos vídeos marcados juntos.

Os comandos abaixo são os que geraram os pesos do sistema. Os scripts acham as pastas pela própria localização, e o que eles geram vai para `treinamento/saidas/`.

| Passo | Comando | Precisa de | Gera |
|---|---|---|---|
| 1. Base pública | `python contador/woinoski_dados.py <pasta da base>` | a base de Woinoski e Bajić, com `rotulos/textFiles/*_str.txt` e `videos/*.avi` | `saidas/woinoski_48.pkl` |
| 2. Pré-treino | `python contador/rede2.py w_T24_3000 3000 24 w` | passo 1 | `saidas/rede2/w_T24_3000.pt` |
| 3. Movimento da câmera | `python contador/movimento_fundo.py <vídeos>` | os vídeos de desenvolvimento | `saidas/fundo/` |
| 4. Recortes dos nadadores | `python contador/extrai_recortes.py <medida>` e depois `python contador/extrai_recortes48.py <medida> <pasta dos vídeos>` | os vídeos e as detecções guardadas de cada medida, em `base_teste/` | `saidas/recortes/` e `saidas/recortes48/` |
| 5. Dados de treino | `python contador/rede_dados.py 48 _ouro3` | passos 3 e 4 e as marcações de [`../marcacoes_braçadas`](../marcacoes_braçadas) | `saidas/rede_dados_48_ouro3.pkl` |
| 6. Treino final | `python contador/rede2.py final_T24_ouro3_ft500 500 24 tudo inicial=w_T24_3000.pt dados=rede_dados_48_ouro3.pkl` | passos 2 e 5 | `saidas/rede2/final_T24_ouro3_ft500.pt`, que é o `sistema_final/contador_rede_pesos.pt` |

Detalhes dos passos:

- **Passo 4.** Uma "medida" é um detector rodado nos vídeos, com as caixas guardadas em `base_teste/resultados/<medida>/cache_deteccoes/`. O `extrai_recortes.py` roda sobre essas caixas o rastreamento da seção 7 do notebook [`rastreamento_usado_nos_recortes.ipynb`](rastreamento_usado_nos_recortes.ipynb). O script lê o notebook sozinho e roda só as seções 1, 2 e 7; não é preciso abri-lo, e o resto do notebook não faz parte do treino. Como a célula de ambiente do notebook roda junto, `base_teste/` precisa ter também `Dataset/dataset nadador Yolo11.zip` e a pasta `videos/`. Os nomes das medidas que o passo 5 usa estão em `contador/rede_dados.py` e em `contador/bancada_contador.py` (`DETECTORES`).
- **Passo 5.** Nos seis vídeos mais antigos, as trilhas de cada detector listado em `DETECTORES` entram como exemplos separados. Nos vídeos do YouTube e do iStock, entram as trilhas do detector final. A raia de cada trilha nas tomadas do YouTube está em `contador/raias_tomadas.py`.
- **Passo 6.** O quarto argumento do `rede2.py` é o modo: `w` treina só com a base pública (é o pré-treino), `tudo` treina com tudo (é o modelo final), `w+n` deixa de fora uma prova nossa (`fora=<grupo>`) e mede nela, e `lovo` repete isso para cada prova, uma de cada vez (é a medida de 1,29 braçada). Vídeos da mesma prova saem do treino juntos (`contador/provas.py`).
- **Passo 2.** O pré-treino rodou em 13/11/2026, com uma versão anterior do `rede2.py`, e não foi repetido com esta. O treino final parte do arquivo que ele gerou.

## Como foi conferido

Com os arquivos desta pasta, os passos 5 e 6 foram refeitos em CPU, num notebook sem GPU. O arquivo de dados saiu igual, byte a byte, ao que treinou a rede. O treino final, de 6 minutos, deu exatamente os pesos do sistema (`contador_rede_pesos.pt`, SHA-1 `a68ec2226554d402089121daa5e6f046aa701734`). Os passos 1 a 4 não foram refeitos, porque precisam da base pública e dos vídeos; os arquivos que eles tinham gerado foram usados como estavam.

O treino do detector não foi refeito, porque precisa de GPU. O que foi conferido é que as células de código do notebook são as mesmas que rodaram, e os parâmetros gravados nos pesos são os descritos acima.
