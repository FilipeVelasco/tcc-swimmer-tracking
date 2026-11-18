# Sistema final: um comando para detectar, rastrear e contar braçadas

Esta pasta roda o sistema num vídeo de natação. Ela acha os nadadores (YOLO11m), segue um nadador por raia (ByteTrack e uma etapa que agrupa as trilhas em raias) e conta as braçadas de cada um. A contagem é a da rede treinada; quando o ritmo que a rede encontra sai irregular, vale a contagem pelo respingo na água.

Os pesos dos dois modelos já estão aqui. O código que os treinou está em [`../treinamento`](../treinamento).

## Como rodar

```
python roda_video.py VIDEO --nadadores N --nado crawl|borboleta
```

Por exemplo:

```
python roda_video.py prova.mp4 --nadadores 8 --nado borboleta
```

| Entrada | O que é |
|---|---|
| `VIDEO` | O vídeo, uma tomada só, sem corte de câmera. Se houver vários cortes, separe um arquivo por tomada. |
| `--nadadores N` | Quantos nadadores aparecem, um por raia. Quem roda informa; o sistema não detecta. |
| `--nado` | `crawl` ou `borboleta`. Quem roda informa; o sistema não reconhece o nado. Outro nado é recusado. |
| `--saida PASTA` | Onde gravar. Padrão: `saidas/<nome do vídeo>`, dentro desta pasta. |
| `--detector CAMINHO` | Pesos do detector. Padrão: `yolo11m_640_final_14_competicoes.pt`, desta pasta. |
| `--contador CAMINHO` | Pesos da rede de contagem. Padrão: `contador_rede_pesos.pt`, desta pasta. |
| `--sem-video` | Não grava o vídeo anotado. |
| `--so-deteccao` | Roda só o detector e para. A detecção fica guardada para a próxima vez. |
| `--refaz-deteccao` | Ignora a detecção guardada e roda o detector de novo. |

O comando começa dizendo o vídeo, onde o detector vai rodar (`cpu` ou o nome da placa) e se os pesos são os que foram medidos:

```
vídeo: prova.mp4 | 169 quadros, 29.97 q/s, 1920x1080 | dispositivo: cpu
pesos do detector: os da medida (yolo11m_640_final_14_competicoes.pt; é o last.pt e o modelo_final.pt do treino final)
pesos do contador: os da medida (contador_rede_pesos.pt)
```

Se aparecer `AVISO: os pesos ... não são os da medida`, os números do trabalho não valem para aquela execução.

## Instalação

1. Python 3.12 ou 3.13.
2. O `torch` indicado para a sua máquina, em pytorch.org.
3. `python -m pip install -r requirements.txt`

O `requirements.txt` traz as versões com que o sistema foi medido. Uma instalação do zero a partir dele não foi testada.

Os pesos do detector são o `last.pt` do treino final. Esse treino deixou dois arquivos com os mesmos pesos e SHA-1 diferentes (só muda o nome de uma pasta dentro do arquivo), e o comando aceita os dois:

| Arquivo do treino | SHA-1 |
|---|---|
| `last.pt`, o que está aqui | `8efbe3803b0d9232f4356b1c3388d62a0522ff31` |
| `best.pt` | `870abf365d3054c941ef42bcf665f129eb682785` |

## O que sai

| Arquivo | Conteúdo |
|---|---|
| `bracadas.csv` | Uma linha por raia, com a contagem. É a tabela principal. |
| `instantes.csv` | O instante de cada braçada contada, por raia. |
| `trilhas.csv` | A caixa de cada raia em cada quadro. O `trilhas.npz` tem as mesmas caixas, sem arredondar. |
| `resumo.json` | Tudo isso, mais os parâmetros, o SHA-1 dos pesos e do código, as versões das bibliotecas, os avisos e o tempo de cada etapa. |
| `<vídeo>_anotado.mp4` | O vídeo com a caixa, o número da raia e a contagem correndo. |
| `deteccoes.csv`, `deteccoes.json` | As caixas do detector, quadro a quadro, e de que vídeo e pesos elas vieram. |

O detector é a etapa demorada, então a detecção fica guardada. Se o vídeo, os pesos e o `rastreamento.py` não mudarem, rodar de novo leva segundos.

Colunas de `bracadas.csv`:

| Coluna | O que é |
|---|---|
| `raia` | Número da linha (veja a nota abaixo). |
| `cobertura_pct` | Em quantos por cento dos quadros a raia tem caixa. |
| `rede` | Contagem da rede. |
| `irregularidade_do_ritmo` | Desvio dividido pela média dos intervalos entre as braçadas da rede. Fica vazia quando a rede acha menos de 4 braçadas. |
| `respingo` | Contagem pelo respingo na região da cabeça. |
| `usou` | `rede` se a irregularidade for até 0,20; senão, `respingo`. |
| `bracadas` | A contagem final. |
| `centro_x`, `centro_y` | Centro mediano da caixa do nadador, em pixels. |

No crawl cada braço conta uma braçada. Na borboleta, cada ciclo.

Três cuidados ao ler:

- **O número da raia pode não ser o da piscina.** É a ordem dos nadadores ao longo do eixo que o sistema escolhe (`eixo_das_raias` no `resumo.json`): no `x`, da esquerda para a direita; no `y`, de cima para baixo. Vale o eixo que aproveita mais caixas e, no empate, o `x`. Para saber quem é quem, use `centro_x` e `centro_y` ou o número desenhado no vídeo anotado.
- **Quando vale o respingo, a contagem pode terminar em meio** (7,5, por exemplo), porque é a média de duas variantes. Os instantes gravados são os de uma delas. O total que vale é a coluna `bracadas`.
- **Raia sem trilha sai com contagem 0, e esse zero não vale.** O comando avisa quais são.

## Limites

- Só crawl e borboleta. Costas e peito não foram validados.
- O contador foi validado em velocidade real. Em câmera lenta, ele erra.
- O número de nadadores e o nado são informados por quem roda.
- Cada tomada de câmera é um vídeo.
- A rede de contagem foi treinada com todos os vídeos marcados. A contagem que este comando dá num desses vídeos não serve como medida de desempenho.

## Como foi o resultado

Os números estão no texto do TCC, com as ressalvas de cada um:

- Nos 79 nadadores dos vídeos de desenvolvimento, o erro médio é de 1,29 braçada quando a prova medida fica fora do treino da rede, e de 1,41 quando a competição inteira fica fora.
- No teste cego, o sistema rodou uma única vez, sem ajuste, em 4 trechos de finais de Paris 2024 que nunca tinha visto. Foram 32 nadadores, marcados antes de o sistema rodar (marcas em [`../marcacoes_braçadas/teste_cego_paris2024`](../marcacoes_braçadas/teste_cego_paris2024)). O erro médio foi de 0,98 braçada por nadador. Ao rever as marcas, o anotador retirou 4 braçadas cortadas pelo fim do vídeo, e o erro passou a 0,86.
- Um primeiro teste cego, com uma versão anterior da etapa de raias, deu 2,75 braçadas numa prova inteira e 0,75 em vídeos de um nadador.
- Duas pessoas contando os mesmos 33 nadadores discordam, em média, 0,46 braçada.

## Arquivos

| Arquivo | O que é |
|---|---|
| `roda_video.py` | O comando: lê os argumentos, chama as etapas na ordem e grava as saídas. |
| `rastreamento.py` | Detector, ByteTrack, trilha de um nadador só e os parâmetros. |
| `agrupa_raias.py` | A etapa que agrupa as trilhas em raias, um nadador por raia. |
| `contador_respingo.py` | A contagem pelo respingo. |
| `contador_rede.py` | A contagem pela rede e a regra que escolhe entre as duas. |
| `contador_rede_pesos.pt` | Os pesos da rede de contagem (SHA-1 `a68ec2226554d402089121daa5e6f046aa701734`). |
| `yolo11m_640_final_14_competicoes.pt` | Os pesos do detector (SHA-1 `8efbe3803b0d9232f4356b1c3388d62a0522ff31`). |
| `requirements.txt` | As versões das bibliotecas da medida. |

As funções de `rastreamento.py`, `agrupa_raias.py` e dos dois contadores vêm do notebook com que o sistema foi medido. Na preparação para a publicação, só os comentários e os textos dos avisos foram revistos, e a conferência abaixo mostra que o resultado não mudou.

## Como foi conferido

Em 15/11/2026, com o critério escrito antes de rodar, o comando rodou nos 13 vídeos de desenvolvimento. Ele deu as mesmas trilhas e as mesmas contagens do notebook em todos os 79 nadadores, e a detecção refeita deu as mesmas caixas guardadas.

Depois, os comentários foram revisados para a publicação e as versões antigas da etapa de raias saíram. Em 17/11/2026 a conferência foi refeita com as detecções guardadas. Nos 13 vídeos (79 nadadores), as contagens, os instantes e as trilhas saíram iguais aos do comando original; as tabelas, byte a byte.

O comando rodou só em CPU e em Python 3.12. O notebook, que tem as mesmas funções, rodou também em GPU e em Python 3.13 e deu as mesmas tabelas. A conferência mostra que o comando repete o notebook. Ela não mede desempenho.

## Quanto tempo leva

Medido num notebook sem GPU (Intel i5-1135G7), nos 13 vídeos de desenvolvimento:

- O detector leva 0,30 s por quadro (3,4 quadros por segundo), cerca de 95% do tempo.
- Sem contar o vídeo anotado, as etapas levam cerca de 9 vezes a duração do vídeo (10,8 vezes no pior vídeo).
- Sem o detector, as outras etapas somadas levam cerca de 0,4 vez a duração do vídeo.

Em GPU, o tempo deste comando não foi medido.
