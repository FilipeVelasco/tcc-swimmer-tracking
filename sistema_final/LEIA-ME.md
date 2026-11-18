# Sistema final: um comando para detectar, rastrear e contar braçadas

Esta pasta roda o sistema num vídeo de natação. Ela acha os nadadores (YOLO11m), segue um nadador por raia (ByteTrack e uma etapa que agrupa as trilhas em raias) e conta as braçadas de cada um. A contagem é a da rede treinada; quando o ritmo que a rede encontra sai irregular, vale a contagem pelo respingo na água.

Os pesos dos dois modelos já estão aqui. O código que os treinou está em [`../treinamento`](../treinamento).

## Como rodar

```
python roda_video.py VIDEO --nado crawl|borboleta [--nadadores N]
```

Por exemplo:

```
python roda_video.py prova.mp4 --nado borboleta
python roda_video.py prova.mp4 --nado borboleta --nadadores 8
```

| Entrada | O que é |
|---|---|
| `VIDEO` | O vídeo. Se tiver troca de câmera, cada tomada é contada separada (veja abaixo). |
| `--nado` | `crawl` ou `borboleta`. Quem roda informa; o sistema não reconhece o nado. Outro nado é recusado. |
| `--nadadores N` | Quantos nadadores aparecem, um por raia. Sem ele, o número é estimado (veja abaixo). |
| `--saida PASTA` | Onde gravar. Padrão: `saidas/<nome do vídeo>`, dentro desta pasta. |
| `--detector CAMINHO` | Pesos do detector. Padrão: `yolo11m_640_final_14_competicoes.pt`, desta pasta. |
| `--contador CAMINHO` | Pesos da rede de contagem. Padrão: `contador_rede_pesos.pt`, desta pasta. |
| `--sem-video` | Não grava o vídeo anotado. |
| `--so-deteccao` | Roda só o detector e para. A detecção fica guardada para a próxima vez. |
| `--refaz-deteccao` | Ignora a detecção guardada e roda o detector de novo. |
| `--tomada-unica` | Não procura troca de câmera: o vídeo inteiro é tratado como uma tomada. |

O comando começa dizendo o vídeo, onde o detector vai rodar (`cpu` ou o nome da placa) e se os pesos são os que foram medidos:

```
vídeo: prova.mp4 | 169 quadros, 29.97 q/s, 1920x1080 | dispositivo: cpu
pesos do detector: os da medida (yolo11m_640_final_14_competicoes.pt; é o last.pt e o modelo_final.pt do treino final)
pesos do contador: os da medida (contador_rede_pesos.pt)
```

Se aparecer `AVISO: os pesos ... não são os da medida`, os números do trabalho não valem para aquela execução.

### Número de nadadores

Sem `--nadadores`, o número é a quantidade de caixas, com confiança de pelo menos 0,25, que o detector acha na maior parte dos quadros (a mediana entre os quadros). O comando diz o número estimado. Confira no vídeo: se estiver errado, rode de novo com `--nadadores N`, e a detecção guardada é reaproveitada.

A estimativa erra quando um nadador fica escondido na maior parte do vídeo. Nos 13 vídeos de desenvolvimento, ela acertou 12. No que errou, o nadador da raia mais distante da câmera, junto da parede e dentro do respingo, aparece em menos da metade dos quadros, e a estimativa deu 8 em vez de 9.

Quando o número é informado e o detector acha outro, o comando avisa.

### Troca de câmera

Antes de contar, o comando procura troca de câmera: uma mudança brusca de um quadro para o seguinte, na cor ou na forma da imagem, seguida de imagem estável. Algo passando na frente da câmera ou a espuma da largada mudam a imagem por vários quadros seguidos e não contam como troca. Transição gradual (uma imagem se fundindo na outra) não é percebida.

Com mais de um nadador, cada tomada é contada separada:

- cada tomada vira um vídeo em `tomadas/`, gravado sem perda (os quadros são os mesmos do original);
- a detecção do vídeo inteiro é repartida entre as tomadas, sem rodar o detector de novo;
- cada tomada tem sua pasta, `tomada_01`, `tomada_02`..., com as mesmas saídas de um vídeo comum;
- tomadas de menos de 2 s ficam de fora;
- sem `--nadadores`, o número é estimado em cada tomada.

**O número da raia vale dentro de cada tomada.** O sistema não sabe se a raia 3 de uma tomada é o mesmo nadador da raia 3 de outra, porque o ângulo muda e a ordem dos nadadores na imagem muda com ele.

Com um nadador só, não há quem confundir: o vídeo é contado inteiro, e o comando só diz onde estão as trocas.

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

| `cortes.json` | Os quadros em que a câmera troca. Guardado para não reler o vídeo na próxima vez. |

O detector é a etapa demorada, então a detecção fica guardada. Se o vídeo, os pesos e o `rastreamento.py` não mudarem, rodar de novo leva segundos.

Quando o vídeo tem troca de câmera e mais de um nadador, a pasta de saída tem uma pasta por tomada e mais:

| Arquivo | Conteúdo |
|---|---|
| `tomadas.csv` | Uma linha por tomada: início, fim, se foi contada, quantos nadadores e se o número foi informado ou estimado. |
| `bracadas_por_tomada.csv` | O `bracadas.csv` de todas as tomadas juntas, com a coluna `tomada`. |
| `instantes_por_tomada.csv` | Os instantes de todas as tomadas, no tempo da tomada e no tempo do vídeo. |
| `resumo.json` | As trocas de câmera e a lista das tomadas. |

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
- O nado é informado por quem roda.
- O número de nadadores estimado pode sair errado quando um nadador fica escondido na maior parte do vídeo. Ele deve ser conferido.
- Com troca de câmera, cada tomada é contada à parte, e o mesmo nadador não é reconhecido de uma tomada para outra. Transição gradual entre câmeras não é percebida.
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
| `tomadas.py` | A procura de troca de câmera, a separação das tomadas e a estimativa do número de nadadores. |
| `rastreamento.py` | Detector, ByteTrack, trilha de um nadador só e os parâmetros. |
| `agrupa_raias.py` | A etapa que agrupa as trilhas em raias, um nadador por raia. |
| `contador_respingo.py` | A contagem pelo respingo. |
| `contador_rede.py` | A contagem pela rede e a regra que escolhe entre as duas. |
| `contador_rede_pesos.pt` | Os pesos da rede de contagem (SHA-1 `a68ec2226554d402089121daa5e6f046aa701734`). |
| `yolo11m_640_final_14_competicoes.pt` | Os pesos do detector (SHA-1 `8efbe3803b0d9232f4356b1c3388d62a0522ff31`). |
| `requirements.txt` | As versões das bibliotecas da medida. |

As funções de `rastreamento.py`, `agrupa_raias.py` e dos dois contadores vêm do notebook com que o sistema foi medido. Na preparação para a publicação, só os comentários e os textos dos avisos foram revistos, e a conferência abaixo mostra que o resultado não mudou.

## Como foi conferido

Com o critério escrito antes de rodar, o comando rodou nos 13 vídeos de desenvolvimento. Ele deu as mesmas trilhas e as mesmas contagens do notebook em todos os 79 nadadores, e a detecção refeita deu as mesmas caixas guardadas.

Depois, os comentários foram revisados para a publicação e as versões antigas da etapa de raias saíram. Após isso, a conferência foi refeita com as detecções guardadas. Nos 13 vídeos (79 nadadores), as contagens, os instantes e as trilhas saíram iguais aos do comando original; as tabelas, byte a byte.

Em seguinte entraram a estimativa do número de nadadores e a troca de câmera, e a conferência foi refeita nos mesmos 13 vídeos:

- com o número de nadadores informado, as saídas continuaram iguais às do comando original nos 79 nadadores;
- sem informar o número, a estimativa acertou 12 dos 13 vídeos, e nesses 12 as saídas também saíram iguais;
- a troca de câmera foi procurada em 25 vídeos de uma tomada só, entre eles os dos dois testes cegos, e em 4 vídeos de 25 s com trocas. Achou as 15 trocas e nenhuma troca falsa. Um dos vídeos de desenvolvimento, de um nadador só, tem 4 trocas de verdade, e o comando o conta inteiro, como antes.

A regra da troca de câmera foi ajustada olhando esses mesmos vídeos (a exigência de imagem estável depois da troca entrou para descartar a espuma da largada e algo passando na frente da câmera), então esses números não medem como ela se sai em vídeos novos.

O comando rodou só em CPU e em Python 3.12. O notebook, que tem as mesmas funções, rodou também em GPU e em Python 3.13 e deu as mesmas tabelas. A conferência mostra que o comando repete o notebook. Ela não mede desempenho.

## Quanto tempo leva

Medido num notebook sem GPU (Intel i5-1135G7), nos 13 vídeos de desenvolvimento:

- O detector leva 0,30 s por quadro (3,4 quadros por segundo), cerca de 95% do tempo.
- Sem contar o vídeo anotado, as etapas levam cerca de 9 vezes a duração do vídeo (10,8 vezes no pior vídeo).
- Sem o detector, as outras etapas somadas levam cerca de 0,4 vez a duração do vídeo.

Em GPU, o tempo deste comando não foi medido.
