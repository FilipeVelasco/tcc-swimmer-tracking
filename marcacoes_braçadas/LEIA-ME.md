# Marcações de braçadas

Braçadas marcadas à mão, vendo o vídeo, usadas como gabarito na avaliação da contagem. Os vídeos não estão no repositório.

Todos os arquivos têm as colunas `video,raia,tempo_s`: o vídeo marcado, a raia e o instante da braçada, em segundos desde o começo do vídeo. No crawl cada braço conta uma braçada; na borboleta, cada ciclo.

O número da raia é o da marcação de cada vídeo, e não necessariamente o da raia da piscina.

## Vídeos de desenvolvimento e outros clipes (esta pasta)

| Arquivos | O que são |
|---|---|
| `bracadas_natacao_teste_*.csv`, `bracadas_yt_*.csv` | Vídeos de prova usados no desenvolvimento. |
| `bracadas_natacao_teste_3_c2_marcacao2.csv` | Uma segunda marcação do mesmo vídeo do `bracadas_natacao_teste_3_c2.csv`. |
| `bracadas_istockphoto-*.csv`, `bracadas_pexels_6539556.csv` | Clipes de um nadador. |
| `bracadas_pexels_6539556_x4.csv` | O mesmo clipe, na versão acelerada 4 vezes (o original é em câmera lenta). |
| `bracadas_1_pexels_6539551.csv`, `bracadas_2_pexels_6539602.csv`, `bracadas_3_natacao_teste_6_c1.csv` | Primeiro teste cego: marcados antes de o sistema rodar nesses vídeos. |
| `bracadas_correcoes_14-11.csv` | Duas braçadas acrescentadas depois da conferência das marcações, com a origem de cada uma. Uma delas tem instante estimado, e o arquivo diz como. |

## `segunda_marcacao_braçadas/`

Uma segunda pessoa marcou 33 nadadores de quatro tomadas sem ver as marcações da primeira. Serve para medir quanto duas pessoas discordam.

Só contam as marcas dentro destas janelas, em segundos do vídeo:

| Tomada | Arquivo | Janela | Nado | Nadadores |
|---|---|---|---|---|
| `natacao_teste_4_c3` | `bracadas_natacao_teste_4_c3_filipe.csv` | 0 a 11,0 | crawl | 8 |
| `yt_2`, tomada 3 | `bracadas_yt_2_filipe.csv` | 33,69 a 42,00 | crawl | 9 |
| `yt_3`, tomada 1 | `bracadas_yt_3_filipe.csv` | 11,645 a 17,284 | borboleta | 8 |
| `yt_3`, tomada 2 | `bracadas_yt_3_filipe.csv` | 22,556 a 28,929 | borboleta | 8 |

Resultado: as duas pessoas discordam em média 0,46 braçada por nadador; contaram igual em 20 dos 33 e diferiram em até 1 braçada em 31.

## `teste_cego_paris2024/`

O teste cego do sistema final, em 4 trechos de finais de natação dos Jogos de Paris 2024 (transmissão da CazéTV no YouTube). Os trechos foram escolhidos por uma regra escrita antes de abrir os vídeos: a primeira tomada contínua em plano aberto, com as 8 raias, depois da saída, com 6 s ou mais, em velocidade real, até 10 s. As braçadas foram marcadas antes de o sistema rodar, e o sistema rodou uma única vez.

| Trecho (coluna `video`) | Prova | Nado | Vídeo do YouTube | Trecho na transmissão (s) | Duração |
|---|---|---|---|---|---|
| `cego2_t1_100livreF` | 100 m livre feminino | crawl | `eEEkPLFc_Vw` | 2.101,583 a 2.109,457 | 7,89 s |
| `cego2_t2_1500livreF` | 1.500 m livre feminino | crawl | `eEEkPLFc_Vw` | 4.699,011 a 4.709,004 | 10,01 s |
| `cego2_t3_100borboletaM` | 100 m borboleta masculino | borboleta | `pUWYchEhZ5U` | 2.320,602 a 2.327,158 | 6,57 s |
| `cego2_t4_800livreF` | 800 m livre feminino | crawl | `pUWYchEhZ5U` | 5.787,865 a 5.796,107 | 8,26 s |

- O tempo das marcas conta a partir do primeiro quadro do trecho.
- A raia 1 é a de cima da imagem, e a 8, a de baixo.
- O vídeo foi baixado a 59,94 quadros por segundo. A medida principal usou um quadro a cada dois (29,97 por segundo).

| Arquivo | O que é |
|---|---|
| `bracadas_teste_cego.csv` | As marcas usadas na execução: 360 braçadas de 32 nadadores. Antes da execução saíram 11 marcas repetidas pelo teclado (tecla segurada), a menos de 0,02 s da anterior na mesma raia. |
| `bracadas_teste_cego_revistas.csv` | As mesmas marcas sem 4 braçadas cortadas pelo fim do vídeo, retiradas pelo anotador ao rever as marcas depois da execução: trecho 1, raias 1 e 6 (7,891 s); trecho 3, raia 2 (6,573 s); trecho 4, raia 6 (8,115 s). |

Resultado: erro médio de 0,98 braçada por nadador com as marcas usadas na execução, e de 0,86 com as marcas revistas.
