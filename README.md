# Contando braçadas em vídeos de natação

Quantas braçadas cada nadador deu naquele trecho da prova? Técnicos contam isso à mão, olhando o vídeo, nadador por nadador. Este projeto tenta fazer a conta sozinho: recebe o vídeo de uma prova, encontra os nadadores, acompanha cada um na sua raia e conta as braçadas.

Ele é o nosso Trabalho de Conclusão de Curso em Ciência da Computação no CEFET/RJ, feito por João Bello Accioly e Filipe Serra Velasco.

## Como funciona

O sistema faz três coisas, uma depois da outra:

1. **Acha os nadadores.** Um detector YOLO11m, que treinamos com imagens de várias competições, marca uma caixa em volta de cada nadador em cada quadro.
2. **Acompanha cada um na sua raia.** O ByteTrack liga as caixas de um quadro ao outro, e uma regra simples garante um nadador por raia.
3. **Conta as braçadas.** Uma rede pequena olha o movimento dentro da caixa de cada nadador e marca o instante de cada braçada. Quando o ritmo que ela encontra sai irregular, vale a contagem pelo respingo na água.

No crawl cada braço conta uma braçada. Na borboleta, cada ciclo.

## Como rodar

```
cd sistema_final
pip install -r requirements.txt   # antes, instale o torch indicado para a sua máquina (pytorch.org)
python roda_video.py prova.mp4 --nadadores 8 --nado crawl
```

O resultado sai em `sistema_final/saidas/`: uma tabela com a contagem de cada raia, o instante de cada braçada e o vídeo com as caixas e a contagem desenhadas. Os detalhes estão no [LEIA-ME do sistema](sistema_final/LEIA-ME.md).

## Como foi o resultado

| O que medimos | Resultado |
|---|---|
| Achar os nadadores (mAP@0,5, cada competição avaliada por um modelo que não a viu) | 0,700 |
| Seguir os nadadores sem trocar um pelo outro (IDF1, num vídeo de 8 nadadores) | 97,9 |
| Contar as braçadas, nos vídeos usados no desenvolvimento | erro médio de 1,29 braçada por nadador |
| Contar as braçadas em provas que o sistema nunca tinha visto (Paris 2024, 32 nadadores) | erro médio de 0,98 braçada por nadador* |
| Para comparar: duas pessoas contando os mesmos nadadores | diferença média de 0,46 braçada |

\* Rodamos o sistema uma única vez nesses vídeos, sem nenhum ajuste depois. Ao rever as marcas, retiramos 4 braçadas que o fim do vídeo cortou pela metade, e o erro passou a 0,86.

O sistema erra mais quando o desenho da transmissão (bandeiras, velocidades, nomes) cobre a raia, e na braçada que fica cortada no fim do vídeo.

## O que tem aqui

| Pasta | O que é |
|---|---|
| [`sistema_final/`](sistema_final/LEIA-ME.md) | O sistema, com os pesos do detector e da rede de contagem. |
| [`treinamento/`](treinamento/LEIA-ME.md) | O código que treinou o detector e a rede de contagem. |
| [`marcacoes_braçadas/`](marcacoes_braçadas/LEIA-ME.md) | As braçadas que marcamos à mão, vendo os vídeos. São o gabarito de tudo o que medimos. |

Os vídeos não estão aqui, porque são de transmissões e de bancos de imagens. O LEIA-ME das marcações diz o que é cada arquivo.

A primeira versão do projeto, que tentava contar as braçadas pela estimativa de pose, foi substituída por este sistema. Ela continua no histórico do repositório, no commit `f57efbc`.

## Dados de outros autores

- O detector foi treinado com o conjunto *Swimmer-detection*, versão 1, do Roboflow Universe, sob licença CC BY 4.0: <https://universe.roboflow.com/swimmingdetection/swimmer-detection-gjwyk/dataset/1>.
- A rede de contagem foi pré-treinada com a base pública de T. Woinoski e I. V. Bajić, do artigo *Swimmer Stroke Rate Estimation From Overhead Race Video* (2021): <https://arxiv.org/abs/2104.12056>.

## Limites

- Só crawl e borboleta.
- Quem roda informa o nado e o número de nadadores.
- Cada vídeo precisa ser uma tomada só, sem troca de câmera, em velocidade real.
- Os números acima vêm de poucos vídeos. O texto do TCC traz cada medida com as suas ressalvas.
