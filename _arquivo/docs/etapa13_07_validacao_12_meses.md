# Etapa 13.7 — Validação da previsão de florações para os próximos 12 meses

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage`. Nada foi enviado ao GitHub.

**Modelo:** `baseline-hist-api-v1`, o oficial desde a 13.3. Na Etapa 13.7 (GLORYS), a física acrescentou só +0,003 de PR-AUC, então ele continua sendo o melhor modelo. Foi usado o modelo **salvo**, sem retreino.

**Como foi feito:** script `etapa13_07_validacao_12_meses.py`, com 4 min de máquina. Saídas em `resultados/etapa13_07_validacao/`.

**Classificação: HINDCAST VALIDADO, não SISTEMA OPERACIONAL** (ver §7).

---

## 1. O que foi avaliado

- **Modelo:** XGBoost multi-horizonte (um modelo para h = 1…12), com base climatológica. Usa o estado de satélite no mês de emissão t0 (CHL, SST, anomalias) e o histórico do ponto e do mês-alvo.
- **Teste:** emissões de 2022-01 a 2026-08 e alvos de 2022-02 a 2026-09. São **1 084 740 previsões** (ponto × emissão × horizonte), todas com alvo observado.
- **Alvo:** floração = CHL máxima mensal > 1,5 mg/m³ (satélite L4, 4 km).
- **Limiares:** escolhidos na validação (2019–2021), nunca no teste. ALTO ≥ 0,39 (máximo F1) e MODERADO ≥ 0,195.
- **Referências:**
  - **climatologia:** frequência histórica de floração naquele ponto e mês-alvo, com anos anteriores ao alvo;
  - **persistência:** floração em t0.

## 2. Saída

O arquivo é `previsoes_teste_hindcast.parquet`, com uma linha por previsão:

| campo | conteúdo |
|---|---|
| prediction_date | mês de emissão t0 (AAAA-MM) |
| target_date | mês-alvo t0 + h |
| horizon | 1…12 |
| latitude, longitude | ponto da grade CHL 4 km |
| probabilidade | P(floração), de 0 a 1 |
| classe | 1 se probabilidade ≥ 0,39 |
| nivel_de_risco | BAIXO < 0,195 ≤ MODERADO < 0,39 ≤ ALTO |
| modelo, versao | XGBoost multi-horizonte com base climatológica; baseline-hist-api-v1 |
| observado, regiao, fonte_inputs | para auditoria |

## 3. Métricas por horizonte (MEDIDO, teste completo)

| h | PR-AUC | ROC-AUC | F1 | recall | precisão | Brier | log loss | PR-AUC climatologia | Δ vs clim [IC 95 %] | PR-AUC persistência |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,792 | 0,912 | 0,678 | 0,596 | 0,786 | 0,080 | 0,267 | 0,783 | +0,009 [−0,006; +0,021] | 0,511 |
| 2 | 0,795 | 0,912 | 0,681 | 0,594 | 0,796 | 0,081 | 0,269 | 0,788 | +0,007 [−0,004; +0,020] | 0,458 |
| 3 | 0,792 | 0,909 | 0,678 | 0,592 | 0,794 | 0,081 | 0,272 | 0,786 | +0,006 [−0,008; +0,019] | 0,409 |
| 4 | 0,788 | 0,906 | 0,674 | 0,588 | 0,790 | 0,082 | 0,274 | 0,783 | +0,005 | 0,390 |
| 5 | 0,784 | 0,903 | 0,672 | 0,585 | 0,787 | 0,083 | 0,278 | 0,779 | +0,005 | 0,372 |
| 6 | 0,780 | 0,902 | 0,668 | 0,583 | 0,783 | 0,083 | 0,279 | 0,775 | +0,005 [−0,008; +0,020] | 0,366 |
| 7 | 0,781 | 0,903 | 0,667 | 0,580 | 0,786 | 0,083 | 0,279 | 0,775 | +0,006 | 0,373 |
| 8 | 0,778 | 0,902 | 0,666 | 0,579 | 0,783 | 0,083 | 0,277 | 0,772 | +0,006 | 0,362 |
| 9 | 0,776 | 0,902 | 0,664 | 0,580 | 0,777 | 0,083 | 0,278 | 0,769 | +0,007 | 0,360 |
| 10 | 0,776 | 0,901 | 0,664 | 0,579 | 0,779 | 0,084 | 0,282 | 0,772 | +0,004 | 0,364 |
| 11 | 0,777 | 0,901 | 0,664 | 0,576 | 0,785 | 0,085 | 0,284 | 0,772 | +0,005 | 0,398 |
| 12 | 0,777 | 0,900 | 0,664 | 0,576 | 0,783 | 0,086 | 0,286 | 0,773 | +0,004 [−0,010; +0,020] | 0,389 |

O IC 95 % é de bootstrap em blocos (ano do alvo × região × trimestre, 300 reamostragens).

**Brier skill contra a climatologia:** de −0,001 (h1) a −0,016 (h12). Na probabilidade, o modelo **não supera** a climatologia.

## 4. Degradação com o horizonte

A qualidade cai **pouco**: PR-AUC de 0,792 em h1 para 0,777 em h12, e F1 de 0,678 para 0,664. **Isso não quer dizer que o modelo prevê bem a longo prazo.** Quer dizer que quase toda a nota vem da climatologia (onde e em que mês costuma florescer), e a climatologia não depende do horizonte. O ganho sobre ela é de +0,004 a +0,009 de PR-AUC, com **IC 95 % cruzando zero em todos os horizontes**.

| horizonte | funciona bem? |
|---|---|
| **H1** | Nota alta (PR-AUC 0,79, ROC 0,91), mas só +0,009 sobre a climatologia, não significativo. **Útil como mapa de risco sazonal; não antecipa anomalias.** |
| **H3** | Igual ao H1 (+0,006, não significativo). |
| **H6** | Igual (+0,005, não significativo). |
| **H12** | PR-AUC 0,777 e Brier pior que a climatologia. **É praticamente a climatologia.** |

A persistência (floração em t0) é muito pior que tudo: 0,51 em h1 e 0,36–0,40 depois.

## 5. Região e localização

**Por setor** (`metricas_por_regiao.csv`):

| região | taxa de floração | PR-AUC h1 / h3 / h6 / h12 | Δ vs climatologia |
|---|---|---|---|
| sul (Peruíbe–Itanhaém–Mongaguá) | 26 % | 0,885 / 0,890 / 0,884 / 0,888 | −0,006 a −0,001: **não ganha** da climatologia |
| centro (Praia Grande–Santos–Guarujá–Bertioga) | 22 % | 0,814 / 0,814 / 0,801 / 0,800 | +0,005 a −0,004 |
| norte (São Sebastião–Ilhabela–Caraguatatuba–Ubatuba) | 14 % | 0,684 / 0,679 / 0,662 / 0,653 | **+0,020 a +0,026**: o único setor com ganho claro |

**Por ponto** (`habilidade_por_ponto.csv`): Brier skill contra a climatologia, mediana e fração de pontos com skill > 0.

| | h1 | h3 | h6 | h12 |
|---|---|---|---|---|
| norte | +0,042 (74 %) | +0,029 (65 %) | +0,031 (67 %) | +0,030 (64 %) |
| centro | +0,006 (56 %) | −0,005 (46 %) | −0,007 (42 %) | −0,027 (26 %) |
| sul | −0,050 (34 %) | −0,049 (32 %) | −0,039 (34 %) | +0,003 (53 %) |

**Localização** (`localizacao_por_horizonte.csv`): em cada mês-alvo, a AUC entre pontos mede se o modelo ordena bem ONDE floresce.
- **Modelo:** 0,952–0,954 em todos os horizontes.
- **Climatologia:** 0,949–0,952.
- **Precisão dos pontos ALTO:** 79–81 % efetivamente floresceram.
- **Leitura:** a localização é confiável porque as florações se repetem nos mesmos lugares (costa, baías, sul). O modelo acrescenta muito pouco à geografia histórica: é melhor que a climatologia em 56–69 % dos meses.

## 6. Calibração

Teste completo (`calibracao.csv`; o padrão se repete em h1, h3, h6 e h12):

| prevista | observada |
|---|---|
| ~20 % (0,192) | **26,4 %** |
| ~50 % (0,496) | **49,1 %** |
| ~80 % (0,802) | **85,0 %** |

Por faixa:

| faixa | prob. média | freq. observada |
|---|---|---|
| 0–10 % | 3,3 % | 4,4 % |
| 10–20 % | 14,0 % | 18,3 % |
| 20–30 % | 24,3 % | 33,2 % |
| 30–40 % | 34,7 % | 41,0 % |
| 40–60 % | ≈ igual | ≈ igual |
| 60–70 % | 64,9 % | 69,0 % |
| 70–80 % | 75,3 % | 79,4 % |
| 80–90 % | 85,2 % | 88,5 % |
| 90–100 % | 96,2 % | 98,7 % |

**Leitura:** 50 % está bem calibrado. Abaixo e acima de 50 %, o modelo **subestima** a frequência real em 3 a 9 pontos percentuais: "20 %" acontece ~26 % das vezes e "80 %" ~85 %. Uma recalibração isotônica na validação corrigiria isso, mas não foi feita nesta etapa.

## 7. Operacional ou hindcast?

**HINDCAST VALIDADO.** Não é SISTEMA OPERACIONAL, por quatro motivos:
- **Inputs reprocessados:** 71 % das linhas de teste (emissões 2022–2024) usam produtos REPROCESSADOS (CHL MY, SST REP), que não existem em tempo real.
- **Inputs de 2025–2026 (29 %):** CHL MY (interino) e SST NRT. É o mais perto do operacional que temos, mas o CHL MY interino também chega com atraso.
  - Nesse subconjunto, o modelo fica **igual ou levemente abaixo** da climatologia: h1 0,808 contra 0,815; h12 0,884 contra 0,884 (`metricas_por_fonte.csv`).
- **Sem prova em tempo real:** nenhuma previsão foi emitida e verificada em tempo real. Não há registro de emissões passadas congeladas antes de o alvo ocorrer.
- **Ingestão sem agendamento:** a ingestão operacional (13.4) existe, mas não tem agendamento.

## 8. Respostas

1. **Quantos horizontes funcionam satisfatoriamente?**
   - **Nota absoluta:** os 12 têm nota alta, PR-AUC 0,78–0,80 e ROC 0,90–0,91.
   - **Ganho sobre a climatologia:** **nenhum** é estatisticamente melhor que a climatologia. O melhor é h1 (+0,009), e todos os IC cruzam zero.
   - **Na prática:** os 12 servem como **mapa de risco sazonal**. Nenhum prevê de forma comprovada anomalias, ou seja, um ano de mais ou menos floração que o normal.
2. **H12 é confiável?** Só no sentido de que "a climatologia é confiável": PR-AUC 0,777, Brier pior que a climatologia, e a habilidade por ponto é negativa em 74 % dos pontos do centro. Como previsão de anomalia, **não**.
3. **A localização é confiável?** **Sim**: AUC espacial de 0,95 e 79–81 % de acerto nos pontos ALTO. Mas isso vem quase todo da geografia histórica das florações, não da previsão.
4. **Quais regiões são melhores ou piores?**
   - **Maior nota:** sul, PR-AUC 0,89, onde floresce mais e de forma mais regular, mas sem ganho sobre a climatologia.
   - **Menor nota:** norte, 0,65–0,68.
   - **Onde o modelo agrega valor:** só no norte (+0,02 a +0,03 sobre a climatologia, com skill > 0 em 64–74 % dos pontos).
   - **Pior relativo:** sul e centro nos horizontes longos.
5. **Qual é a limitação principal?** **O modelo não supera a climatologia.** O estado de satélite em t0 tem pouca memória para meses à frente. A física (GLORYS, 13.6/13.7) acrescentou no máximo +0,003, e a produção CROCO foi descartada pelo custo. Em segundo plano, faltam inputs de tempo real validados e a calibração está levemente subconfiante.
6. **O sistema já pode ir para uma versão operacional?** **Ainda não, como "previsão".** Pode ir como **produto de risco climatológico mensal**, rotulado como tal: "probabilidade histórica ajustada pelo estado atual". Para chamar de operacional, faltam:
   - ingestão NRT agendada;
   - emissões congeladas e verificadas depois do mês-alvo, por pelo menos alguns meses;
   - recalibração isotônica;
   - documentar ao usuário que a habilidade sobre a climatologia não é significativa.
