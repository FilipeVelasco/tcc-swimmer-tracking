# TCC - Deteccao de Nadadores com YOLO11

Este projeto treina um detector de nadadores usando o dataset que ja esta na
pasta `dataset/` e o modelo pesado `YOLO11x`.

## 1. O que precisa ser copiado

Copie a pasta inteira para o computador de treinamento, mantendo esta
estrutura:

```text
treinamento_YOLO11.py
teste_simples.py
requirements.txt
dataset/swimming.v1i.yolov11/
    data.yaml
    train/images/
    train/labels/
    valid/images/
    valid/labels/
    test/images/
    test/labels/
```

O dataset atual esta completo: 3.122 imagens de treino, 893 de validacao e
448 de teste. Nao e necessario baixar outro dataset no Roboflow para este
treinamento.

## 2. Abrir o terminal na pasta do projeto

Entre na pasta que contem `treinamento_YOLO11.py`:

```bash
cd caminho/para/TCC-nadador-main
```

No Windows, tambem e possivel abrir a pasta no Explorer, clicar na barra de
endereco, digitar `cmd` e pressionar Enter.

## 3. Criar o ambiente virtual

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. Instalar as bibliotecas

Com o ambiente virtual ativado:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Em uma maquina com GPU NVIDIA, confirme que o PyTorch instalado esta
compatível com o CUDA dessa maquina antes do treinamento. A GPU deve aparecer
corretamente no `nvidia-smi`.

## 5. Baixar o modelo YOLO11x

Execute este comando na raiz do projeto:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

O arquivo `yolo11x.pt` sera baixado automaticamente. Esse download precisa de
internet e acontece apenas na primeira vez. O arquivo deve ter dezenas ou
centenas de megabytes; um arquivo de poucos bytes e apenas um ponteiro Git
LFS e nao serve como peso.

## 6. Conferir se a GPU esta disponível

Em uma maquina NVIDIA:

```bash
nvidia-smi
```

Se a GPU for a primeira listada, sera usado `--device 0`.

## 7. Iniciar o treinamento pesado

Com a GPU forte pronta:

```bash
python treinamento_YOLO11.py --device 0
```

O treinamento usa:

- `yolo11x.pt`;
- 150 epocas;
- imagens em 1024 pixels;
- batch automatico (`--batch -1`);
- early stopping;
- augmentacao padrao do Ultralytics;
- cache em disco;
- mixed precision;
- graficos e metricas.

Para ajustar conforme a memoria da GPU:

```bash
python treinamento_YOLO11.py --device 0 --batch 8
python treinamento_YOLO11.py --device 0 --imgsz 1280 --batch -1
```

Nao altere a resolucao ou o batch sem necessidade: resolucao maior e batch
maior exigem mais memoria e podem causar `CUDA out of memory`.

## 8. Onde encontrar o modelo treinado

Ao terminar, o melhor modelo estara em:

```text
resultados/treino_yolo11x/weights/best.pt
```

As metricas, graficos e o ultimo checkpoint ficam na mesma pasta de resultado.

## 9. Testar o modelo

Ainda na raiz do projeto:

```bash
python teste_simples.py \
  --input nadadoras02.JPG \
  --modelo resultados/treino_yolo11x/weights/best.pt \
  --device 0
```

Para abrir a imagem anotada na tela:

```bash
python teste_simples.py \
  --input nadadoras02.JPG \
  --modelo resultados/treino_yolo11x/weights/best.pt \
  --device 0 \
  --show
```

O resultado sera salvo dentro de `resultados/`.

## 10. Treinar novamente

O treinamento nao deve ser sobrescrito. Para uma nova tentativa, renomeie a
pasta:

```bash
mv resultados/treino_yolo11x resultados/treino_yolo11x_anterior
```

No Windows, renomeie `resultados/treino_yolo11x` pelo Explorer ou PowerShell.

## Observacao sobre Git LFS

Se os pesos forem obtidos por Git LFS, rode:

```bash
git lfs install
git lfs pull
```

Ao usar pendrive sem internet, copie o arquivo real `yolo11x.pt`, o dataset e
os scripts. Nao e necessario copiar `.venv/`, `resultados/`, `runs/` ou
`__pycache__/`.
