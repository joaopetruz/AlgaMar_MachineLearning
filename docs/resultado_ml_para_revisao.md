# AlgaMar: resultado do modelo de risco de floração (para revisão independente)

Você está recebendo o resultado de um modelo de machine learning para o litoral de São Paulo (Brasil). Avalie se os números fazem sentido, do ponto de vista de oceanografia e de ML. As perguntas estão no fim (seção 8).

Data da previsão: emitida com dados até **set/2026** (execução de 09/10/2026). Hoje: 23/10/2026.

## 1. O que o modelo prevê

- **Alvo ("floração"):** a clorofila-a **máxima diária do mês** num ponto da grade passa de **1,5 mg/m³**. É biomassa de fitoplâncton, **não** alga tóxica. Espécie, toxicidade e contagem de células não existem nos dados.
- **Grade:** 2.406 pontos de 4 km, do litoral de Ubatuba até Cananéia (lat −23,3 a −25,7; lon −44,8 a −48,2), incluindo mar aberto.
- **Entradas:**
  - clorofila-a de satélite (Copernicus L4 gap-free, 4 km, diária);
  - temperatura da superfície (OSTIA L4);
  - salinidade de satélite;
  - climatologia de cada ponto e mês, calculada só com anos anteriores.
  - Usa o mês de emissão e os 2 anteriores.
- **Saída:** a probabilidade de floração em cada um dos 12 meses seguintes (horizontes 1 a 12), num único modelo multi-horizonte.
- **Modelo:** ensemble de 8 XGBoost (`risco-sat-clim-sp-v2-ens-bruto`). Ele parte da climatologia do ponto e mês (em log-odds) e aprende só o desvio.
- **Histórico:** jan/2000 a set/2026.

## 2. Como foi validado

- **Divisão temporal:**
  - treino com alvos até dez/2018;
  - validação de 2019 a 2021, usada para escolher o número de árvores e o limiar;
  - teste com previsões emitidas a partir de jan/2022 e alvos até set/2026, num total de 1.458.036 previsões.
- **Sem vazamento:** as features usam só meses até o mês de emissão. Num teste, todo o painel posterior a mar/2025 foi trocado por números aleatórios, e a previsão emitida em mar/2025 saiu idêntica.

| referência | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| **Modelo** | **0,880** | **0,933** | **0,082** |
| Climatologia (frequência histórica no ponto e mês) | 0,873 | 0,930 | 0,084 |
| Persistência (= mesmo mês do ano anterior) | 0,541 | 0,778 | 0,178 |

Taxa de floração no teste: 27%. Com o limiar ALTO (0,48), o modelo tem recall de 0,70 e precisão de 0,86.

**Calibração no teste** (probabilidade prevista × frequência observada, por faixa):

| faixa | prevista | observada | n |
|---|---|---|---|
| 0–10% | 3,9% | 2,9% | 721.059 |
| 10–20% | 15,0% | 15,0% | 181.863 |
| 20–30% | 24,4% | 22,6% | 117.535 |
| 30–40% | 34,7% | 33,3% | 73.504 |
| 40–50% | 44,8% | 43,6% | 50.625 |
| 50–60% | 54,7% | 55,2% | 33.913 |
| 60–70% | 64,9% | 65,0% | 30.606 |
| 70–80% | 75,0% | 76,0% | 32.227 |
| 80–90% | 85,4% | 88,3% | 45.878 |
| 90–100% | 96,4% | 99,0% | 170.826 |

**Níveis de risco** usados no dashboard: BAIXO abaixo de 24%, MODERADO de 24% a 47%, ALTO de 48% a 79% e MUITO ALTO a partir de 80%.
- 24% e 48% são os limiares escolhidos na validação (o de 48% maximiza o F1).
- 80% foi escolhido pela calibração: acima dele, a floração se confirmou em 88 a 99% dos casos.

## 3. Onde: os 14 pontos (municípios)

Cada município é um ponto de mar em frente à orla. O valor do município é a média dos pontos da grade a até **20 km** desse ponto (de 10 a 46 pontos por município).

| município | região | lat | lon |
|---|---|---|---|
| Ubatuba | Litoral Norte | −23,47 | −44,98 |
| Caraguatatuba | Litoral Norte | −23,66 | −45,36 |
| Ilhabela | Litoral Norte | −23,85 | −45,25 |
| São Sebastião | Litoral Norte | −23,86 | −45,50 |
| Bertioga | Baixada Santista | −23,88 | −46,10 |
| Guarujá | Baixada Santista | −24,03 | −46,22 |
| Santos | Baixada Santista | −24,02 | −46,33 |
| Praia Grande | Baixada Santista | −24,07 | −46,45 |
| Mongaguá | Baixada Santista | −24,14 | −46,61 |
| Itanhaém | Baixada Santista | −24,23 | −46,78 |
| Peruíbe | Baixada Santista | −24,37 | −46,98 |
| Iguape | Litoral Sul | −24,75 | −47,40 |
| Ilha Comprida | Litoral Sul | −24,92 | −47,62 |
| Cananéia | Litoral Sul | −25,10 | −47,85 |

## 4. A previsão: probabilidade de floração (%) por município e mês

Emitida com dados até set/2026, para out/2026 a set/2027.

| município | out/26 | nov | dez | jan/27 | fev | mar | abr | mai | jun | jul | ago | set/27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ubatuba | 47 | 46 | 36 | 32 | 34 | 43 | 47 | 63 | 66 | 72 | 73 | 58 |
| Caraguatatuba | 83 | 74 | 64 | 67 | 75 | 84 | 94 | 97 | 98 | 98 | 97 | 90 |
| Ilhabela | 34 | 42 | 31 | 19 | 27 | 43 | 49 | 65 | 77 | 80 | 78 | 57 |
| São Sebastião | 67 | 63 | 52 | 44 | 66 | 78 | 76 | 80 | 90 | 94 | 94 | 76 |
| Bertioga | 81 | 71 | 57 | 77 | 87 | 93 | 92 | 95 | 97 | 97 | 96 | 87 |
| Guarujá | 64 | 59 | 43 | 59 | 70 | 80 | 80 | 86 | 91 | 87 | 91 | 81 |
| Santos | 86 | 83 | 73 | 84 | 92 | 92 | 93 | 95 | 97 | 96 | 96 | 92 |
| Praia Grande | 77 | 74 | 55 | 71 | 89 | 88 | 89 | 91 | 95 | 94 | 95 | 91 |
| Mongaguá | 75 | 68 | 54 | 65 | 85 | 84 | 84 | 88 | 93 | 91 | 94 | 90 |
| Itanhaém | 79 | 75 | 62 | 76 | 89 | 88 | 86 | 92 | 96 | 94 | 96 | 91 |
| Peruíbe | 84 | 79 | 72 | 86 | 93 | 93 | 92 | 94 | 97 | 96 | 97 | 90 |
| Iguape | 90 | 82 | 83 | 95 | 97 | 97 | 94 | 95 | 96 | 96 | 97 | 91 |
| Ilha Comprida | 87 | 76 | 75 | 87 | 93 | 94 | 90 | 90 | 90 | 92 | 95 | 86 |
| Cananéia | 92 | 81 | 84 | 91 | 95 | 95 | 93 | 94 | 96 | 96 | 96 | 90 |

- **Diferença entre o modelo e a climatologia** nessas 168 células: média de **1,7 ponto percentual** e máxima de **4 pontos**. Na prática, o modelo reproduz a climatologia.
- **Litoral todo** (2.368 pontos, incluindo mar aberto), out/2026: probabilidade média de 23%; 436 pontos com probabilidade ≥ 48%.

## 5. O que foi observado: fração dos pontos com floração (%) nos últimos 12 meses

Mesma definição do alvo, medida por satélite. Não é previsão.

| município | out/25 | nov | dez | jan/26 | fev | mar | abr | mai | jun | jul | ago | set/26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ubatuba | 98 | 48 | 43 | 30 | 37 | 52 | 61 | 91 | 54 | 74 | 67 | 74 |
| Caraguatatuba | 100 | 85 | 95 | 85 | 95 | 100 | 95 | 100 | 100 | 100 | 100 | 100 |
| Ilhabela | 92 | 14 | 32 | 14 | 3 | 27 | 41 | 97 | 65 | 76 | 54 | 70 |
| São Sebastião | 100 | 33 | 78 | 69 | 58 | 100 | 89 | 86 | 89 | 100 | 100 | 75 |
| Bertioga | 89 | 74 | 56 | 93 | 93 | 96 | 93 | 100 | 100 | 100 | 100 | 100 |
| Guarujá | 97 | 48 | 45 | 76 | 55 | 66 | 72 | 69 | 83 | 93 | 93 | 100 |
| Santos | 100 | 80 | 80 | 100 | 90 | 90 | 80 | 90 | 100 | 100 | 90 | 100 |
| Praia Grande | 100 | 53 | 68 | 95 | 84 | 95 | 89 | 89 | 95 | 89 | 79 | 100 |
| Mongaguá | 100 | 39 | 65 | 91 | 87 | 91 | 65 | 78 | 87 | 74 | 100 | 100 |
| Itanhaém | 100 | 63 | 44 | 78 | 100 | 100 | 74 | 93 | 100 | 74 | 96 | 100 |
| Peruíbe | 100 | 84 | 71 | 81 | 87 | 100 | 81 | 97 | 100 | 87 | 100 | 100 |
| Iguape | 100 | 67 | 79 | 91 | 100 | 100 | 98 | 93 | 93 | 91 | 100 | 100 |
| Ilha Comprida | 100 | 54 | 46 | 73 | 71 | 100 | 71 | 93 | 83 | 93 | 100 | 100 |
| Cananéia | 100 | 60 | 84 | 67 | 86 | 100 | 95 | 100 | 98 | 100 | 100 | 100 |

**Condições medidas em set/2026** (média da faixa de 20 km):
- temperatura: 20,3 °C (Cananéia) a 22,0 °C (Ubatuba);
- salinidade: 34,6 PSU (Cananéia) a 35,6 PSU (Ubatuba);
- clorofila média: 1,0 mg/m³ (Ilhabela) a 2,7 mg/m³ (Iguape);
- clorofila máxima: 1,7 a 7,1 mg/m³;
- pH de superfície: 8,10 a 8,12. Esse valor vem do modelo biogeoquímico da Copernicus, não de medição, e não entra no ML.

## 6. Como interpretar (a leitura de quem fez)

1. **O padrão de onde:** o risco cresce de norte para sul e da costa para o mar aberto. Ilhabela e Ubatuba têm o menor risco; Iguape, Cananéia, Ilha Comprida e Santos, o maior. Isso coincide com a influência de estuários (complexo Cananéia–Iguape–Ribeira e estuário de Santos) e de águas costeiras turvas.
2. **O padrão de quando:** o pico é em **maio–agosto** (outono e inverno) e o mínimo em **dez–jan**, em todos os municípios.
3. **O modelo quase repete a climatologia.** Ele ganha dela por +0,007 de PR-AUC, um ganho estatisticamente real mas pequeno. Não antecipa anos atípicos: em out/2025, 52% dos pontos do litoral floresceram, contra 22% previstos (previsão emitida em set/2025).
4. **Perto da costa, "floração" acontece quase todo mês.** Nos últimos 12 meses, vários municípios tiveram 100% dos pontos acima de 1,5 mg/m³. Por isso as probabilidades de costa ficam em 80–97%. O limiar fixo de 1,5 mg/m³ parece baixo para a costa, e isso pode refletir turbidez e não fitoplâncton.

## 7. O que já foi testado e não melhorou

- **Física do oceano (modelo CROCO, Navier–Stokes, 1,5 km):** correntes, vorticidade e velocidade vertical **pioraram** a PR-AUC (−0,055 a −0,061), ficando abaixo até de um controle com os valores embaralhados.
- **Reanálise GLORYS (25 anos):** temperatura e salinidade não acrescentaram nada; a camada de mistura deu +0,003.
- **Retreino com anos recentes** (treino até 2023 e validação 2024–2025): o early stopping escolheu 0 árvores em 7 dos 8 membros. O modelo virou climatologia pura e não foi adotado.

## 8. Perguntas para você (IA revisora)

1. A **sazonalidade prevista** (pico em mai–ago e mínimo em dez–jan na clorofila de superfície do litoral de SP) é coerente com o que se conhece da plataforma sudeste brasileira (pluma do Prata no inverno, ressurgência de Cabo Frio e Ubatuba na primavera e verão, descarga dos rios no verão)?
2. O **gradiente espacial** (Litoral Sul e Santos com mais risco; Ilhabela e Ubatuba com menos) faz sentido?
3. O limiar fixo de **1,5 mg/m³ de clorofila máxima mensal** é adequado para chamar de "floração", sobretudo perto da costa e de estuários, onde o satélite superestima a clorofila em água turva (Case 2)? Seria melhor um limiar relativo, como um percentil do próprio ponto ou uma anomalia?
4. Um modelo que **empata com a climatologia** (PR-AUC 0,880 contra 0,873), mas é **bem calibrado**, é útil como "mapa de risco"? Como você apresentaria isso com honestidade?
5. Há algum sinal de **vazamento ou erro de avaliação** na descrição da seção 2?
6. O que seria preciso para prever **floração de algas tóxicas** (*Dinophysis*, *Alexandrium*, *Gymnodinium catenatum*, *Pseudo-nitzschia*) em vez de clorofila alta?
