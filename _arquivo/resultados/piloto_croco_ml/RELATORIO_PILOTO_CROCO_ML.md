# Piloto exploratório CROCO/Navier–Stokes → ML do AlgaMar

> **Este é um teste piloto exploratório. A série CROCO disponível é insuficiente para concluir que as features físicas melhoram a capacidade de previsão futura do modelo.**

**Data:** 2026-10-09. **Custo:** extração 3 min, análise 10 min, controle extra 1 min. Nenhuma simulação nova foi feita.

**Preservado:** o modelo oficial, o baseline, o alvo, o split, a física do CROCO e o GitHub não foram tocados. Tudo está em `ML/repo/resultados/piloto_croco_ml/`.

**Scripts:**
- `cfd_navier_stokes/scripts/piloto_croco_ml_extrai.py`: CROCO → tabela mensal, com as funções 13.0/13.1;
- `ML/repo/piloto_croco_ml.py`: auditoria, associação, modelos e controles;
- `resultados/piloto_croco_ml/controle_profundidade.py`.

---

## 1. Dados CROCO que existem de fato

Foram inventariados todos os `croco_avg.nc` no WSL: 58 arquivos. A maioria são testes de 1–3 dias de configurações antigas (D1, REST, AGRIF). Só serviram estes:

| fonte | período (médias diárias) | dias | configuração | uso |
|---|---|---|---|---|
| piloto 13.1 (`PILOTO_ML_1301`) | 18/11–30/11/2017 | 13 | **oficial** (build_d2_algamar_final_v1) | principal |
| piloto 13.1 | 01/12–17/12/2017 | 17 | oficial | principal |
| produção interrompida (`producao_ml_v1/runs/201401/parte1`) | 02/01–05/01/2014 | **4** | oficial, só 4 dias após o início a partir do GLORYS (sem spin-up) | principal, com ressalva |
| `GLORYS_ERA5_TIDES_RIOS_jan2024_7d` (12.13 P4) | 02/01–08/01/2024 | 7 | **pré-oficial** (bry com 12 h de atraso, sem EXACT_RESTART) | só sensibilidade |

**Totais:**
- **41 dias**, distribuídos em **4 meses parciais**, nenhum completo;
- 3 meses na configuração oficial e 1 na pré-oficial;
- nenhum mês com mais de 55 % de cobertura.

**Pontos e join** (`cobertura_croco.json`):
- 1 853 pontos da grade CHL, dos quais os **1 790 pontos do ML casaram 100 %**, com chave (lat, lon, ano, mes) e `round(5)`;
- 4 pontos NAO_USAR_ESTUARIO_CANAL foram excluídos.

**Linhas** (ponto × mês de emissão t0, alvo = floração em t0 + 1):
- 2014-01 → 2014-02: 1 712 linhas;
- 2017-11 → 2017-12: 1 786;
- 2017-12 → 2018-01: 1 786;
- 2024-01 → 2024-02: 1 786.
- No total, **5 284 linhas oficiais**, ou 7 070 com a de 2024.

**Prevalência muito diferente entre meses:** 51,6 % de floração em fev/2014, 3,3 % em dez/2017, 7,8 % em jan/2018 e 10,3 % em fev/2024.

**Features com dado válido** (`auditoria_dados_croco.csv`): **todas as 22** dos grupos existem no arquivo.

| % de NaN | features |
|---|---|
| 0 % | T/S de superfície e de fundo, MLD, N², u, v, velocidade, SSH, HBL, transportes |
| < 4 % | divergência, vorticidade, W 10 m |
| 8 % | T/S a 20 m (onde h < 20 m) |
| **35 %** | T/S a 50 m e W 50 m (onde h < 50 m) |

**Nada foi imputado.** Os status de validação foram preservados: TS, CORE e VERTICAL = VALIDADA; CURRENT e DYNAMICS = NAO_VALIDADA_OBS. W **não** foi tratado como upwelling.

## 2. Desenho (sem vazamento)

- **Física:** média CROCO do mês **t0** (parcial).
- **Features do ML:** as 25 do oficial, todas ≤ t0.
- **Alvo:** floração em **t0 + 1** (h = 1). O alvo do próprio t0 foi usado só como diagnóstico.
- **Modelos P0–P4:** XGBoost com os **mesmos hiperparâmetros** do oficial (400 árvores, sem early stopping, porque não há validação) e a mesma base climatológica.
- **Avaliação:** **deixa um mês de fora**, treinando em 2 meses e testando no 3º. Não há CV aleatória.
- **Controle embaralhado:** as colunas CROCO são permutadas dentro de cada mês, 20 vezes.

> **Este teste mede apenas separabilidade/sinal dentro da amostra e não capacidade real de previsão futura.**
> Com 3 meses, a métrica conjunta mistura dois efeitos: o ranking dentro do mês e o "nível" entre meses. Este último tem efetivamente n = 3. Por isso também se reporta a PR-AUC **dentro de cada mês**.

## 3. Associação de cada feature com a floração do mês seguinte

`associacao_features.csv`; correlação de Spearman dentro do mês (média dos 3 meses oficiais).

| feature | grupo | Spearman dentro do mês | Spearman com o resíduo da climatologia | info mútua |
|---|---|---|---|---|
| temperature_surface | TS | **+0,35** | +0,52 | 0,25 |
| hbl | DYNAMICS | **−0,34** | −0,26 | 0,21 |
| salinity_bottom | VERTICAL | −0,27 | −0,35 | 0,17 |
| mld | CORE | −0,26 | −0,22 | 0,18 |
| temperature_bottom | VERTICAL | +0,25 | −0,33 | 0,10 |
| salinity_surface | TS | −0,22 | +0,20 | 0,10 |
| u_east_surface | CURRENT | +0,17 | −0,14 | 0,04 |
| n2_max | CORE | +0,16 | +0,14 | 0,11 |
| v_north / transportes | CURRENT/DYN | +0,12 a +0,14 | −0,05 a −0,10 | 0,05–0,09 |
| vorticidade, ζ/f, divergência, W | DYNAMICS | **|ρ| ≤ 0,05** | ≤ 0,13 | ≤ 0,03 |
| *(referência)* chl_media_t0, satélite | ML | +0,38 | — | — |
| *(referência)* freq_floracao_alvo | ML | +0,45 | — | — |
| *(referência)* GLORYS T sup. | GLORYS | −0,15 | +0,43 | 0,25 |
| *(referência)* GLORYS MLD | GLORYS | −0,18 | −0,24 | 0,16 |

**Leitura (correlação não é causalidade):**
- **T de superfície, HBL e MLD do CROCO** têm associação moderada, até mais forte que a do GLORYS nos mesmos meses. Só que **o sinal não é estável**:
  - **jan/2014 (4 dias):** a T do CROCO correlaciona **ao contrário** do satélite (ρ = −0,33 com a SST de satélite e −0,26 com a do GLORYS). É provavelmente o aquecimento espúrio das áreas rasas abrigadas, já visto na 12.13, logo após o início a frio (INFERÊNCIA).
  - **2017:** CROCO, satélite e GLORYS concordam (ρ ≈ 0,75–0,79).
- **HBL, MLD e salinidade de fundo** também carregam informação de "costa rasa". Mas a profundidade sozinha **não** reproduz o ganho (§5).
- **Correntes, vorticidade, divergência e W:** associação nula a fraca.

## 4. Modelos piloto (3 meses oficiais, deixando um mês de fora)

`metricas_modelos.csv` e `comparacao_modelos.csv`. Δ contra P0 com IC 95 % por bootstrap de blocos (mês × região).

| modelo | PR-AUC conjunta | ROC | F1 | Brier | Δ PR-AUC [IC 95 %] | PR-AUC dentro do mês: 2014-01 / 2017-11 / 2017-12 |
|---|---|---|---|---|---|---|
| **P0 baseline** (features oficiais) | 0,426 | 0,709 | 0,328 | 0,169 | — | 0,806 / 0,653 / 0,889 |
| P1 + CROCO_CORE | **0,483** | 0,770 | 0,338 | 0,164 | **+0,057 [+0,011; +0,090]** | 0,798 / **0,726** / 0,850 |
| Pt + CROCO_TS | 0,458 | 0,740 | 0,327 | 0,167 | +0,032 [+0,003; +0,059] | 0,797 / 0,707 / 0,876 |
| Pv + CROCO_VERTICAL | 0,444 | 0,759 | 0,321 | 0,171 | +0,018 [−0,042; +0,057] | 0,861 / 0,703 / 0,909 |
| P2 + CROCO_CURRENT | 0,371 | 0,677 | 0,296 | 0,180 | **−0,055 [−0,102; −0,015]** | 0,803 / 0,661 / 0,883 |
| P3 + CROCO_DYNAMICS | 0,365 | 0,685 | 0,303 | 0,180 | **−0,061 [−0,123; −0,005]** | 0,813 / 0,561 / 0,908 |
| P4 + todas CROCO | 0,467 | 0,766 | 0,346 | 0,168 | +0,041 [−0,017; +0,091] | 0,797 / 0,678 / 0,893 |
| Pg + GLORYS T/S + MLD | 0,423 | 0,646 | 0,359 | 0,165 | −0,003 [−0,093; +0,190] | 0,805 / 0,658 / 0,870 |

**Sensibilidade com os 4 meses** (inclui jan/2024 pré-oficial): mesmo padrão.
- P1: +0,096 [+0,045; +0,219]; P4: +0,075.
- P2: −0,004; P3: −0,034.
- Pg (GLORYS): −0,022.

**As probabilidades mudaram?** Pouco. A mudança absoluta média contra P0 é de 0,016 (CORE) a 0,021 (todas), com percentil 95 de 0,08–0,13 (`comparacao_modelos.csv`).

**Controle embaralhado** (`controle_embaralhado.csv`, 20 permutações dentro do mês):

| modelo | real | embaralhado (média ± dp, máx) | P0 | p empírico |
|---|---|---|---|---|
| P1 CORE | **0,483** | 0,430 ± 0,003 (0,437) | 0,426 | 0,048 (o mínimo possível com 20) |
| P4 todas | **0,467** | 0,435 ± 0,006 (0,446) | 0,426 | 0,048 |
| P2 CURRENT | 0,371 | 0,424 ± 0,004 | 0,426 | 1,0: **pior que o embaralhado** |
| P3 DYNAMICS | 0,365 | 0,421 ± 0,005 | 0,426 | 1,0: **pior que o embaralhado** |

**Importância** (`feature_importance.csv`, SHAP médio |contrib| do P4):
- **temperature_surface do CROCO é a feature nº 1** (24 % do SHAP total), à frente da SST de satélite (13 %) e da CHL em t0 (8 %);
- depois vêm, entre as CROCO, temperature_20m (4 %), mld (3 %), transport_north (3 %) e ssh (2 %);
- **somando, as CROCO têm cerca de 40 % do SHAP total.**

## 5. Controle extra: é só geografia?

`controle_profundidade.csv`: a profundidade do D2 (h) como única feature extra.

| modelo | conjunta | 2014-01 | 2017-11 | 2017-12 |
|---|---|---|---|---|
| P0 | 0,426 | 0,806 | 0,653 | 0,889 |
| P0 + profundidade | 0,424 | 0,808 | 0,644 | 0,888 |
| P0 + CROCO_CORE | 0,483 | 0,798 | 0,726 | 0,850 |
| P0 + profundidade + CORE | 0,490 | 0,797 | 0,740 | 0,848 |

A profundidade sozinha não muda nada, então o efeito do CORE **não é** apenas "raso × profundo".

**Mas o ganho do CORE vem de um único mês:** dentro do mês, ele **melhora 2017-11 (+0,07)** e **piora 2014-01 (−0,01) e 2017-12 (−0,04)**. A maior parte do ganho conjunto vem de acertar melhor o **nível** entre meses (fev/2014 com 52 % de floração contra dez/2017 com 3 %). Isso é sustentado por só 3 meses.

## 6. Comparação com o GLORYS

- **No GLORYS (13.6/13.7, 25 anos):** MLD deu ganho pequeno; T/S, nada além do satélite; correntes pioraram.
- **Neste piloto:**
  - o CROCO mostra **associação maior** da T de superfície, do HBL e da MLD do que o GLORYS nos mesmos meses;
  - o GLORYS não ajudou nada aqui (Pg ≈ P0);
  - **correntes, vorticidade, divergência, transporte e W não trouxeram sinal** e pioraram o modelo, abaixo até do controle embaralhado, o que é sinal de ruído/sobreajuste.
- **O que isso não permite concluir:** que o CROCO é melhor que o GLORYS. Com 3 meses (um deles de 4 dias sem spin-up), a diferença pode ser acaso, efeito de nível entre meses ou artefato costeiro.

## 7. Classificação

**B — SINAL FRACO.**
- **A favor de sinal:**
  - T/S de superfície e MLD/N² (CORE) superam o baseline (+0,057, IC > 0) e o controle embaralhado no conjunto;
  - T de superfície, HBL e MLD têm associação moderada dentro do mês.
- **Contra:**
  - o ganho **não é estável entre meses** (melhora 1 de 3 e piora 2 de 3 dentro do mês);
  - depende do nível entre só 3 meses;
  - jan/2014 tem só 4 dias e começou sem spin-up;
  - a T do CROCO contradiz o satélite nesse mês.
- **Correntes e dinâmica:** **sem sinal** (pioram).

## 8. Decisão sobre próximos passos

- **Não vale** produzir uma série CROCO longa (2000–2024 ou 2014–2024): o custo é alto e o sinal está restrito ao CORE, e é instável.
- **Se quiser confirmar o sinal, o menor experimento útil:**
  - produzir **só alguns meses completos com spin-up**, por exemplo **6 meses de setembro/outubro de anos diferentes**, onde a física teve o maior ganho no GLORYS (13.6, SON +0,22);
  - repetir **exatamente** este piloto (mesmo script), deixando um mês de fora entre 6–9 meses;
  - **custo:** cada mês ≈ 9–10 h de máquina, em segmentos ≤ 1,5 h com EXACT_RESTART. Total ≈ 60 h, rodável aos poucos, uma execução por vez.
  - **É sua decisão:** cada execução respeita o limite de 1–2 h, mas o total não.
- **Features com prioridade:** temperature_surface, mld, n2_max e hbl.
- **Deixar de fora:** correntes, vorticidade, divergência, W e transportes. Não mostraram sinal e pioram o modelo.
- **Alternativa sem CROCO:** testar a MLD do GLORYS de forma operacional (ganho de +0,003 no oficial) ou parar aqui.

## 9. Respostas

1. **Quantos dados CROCO existiam?** 41 dias em 4 meses parciais (13 + 17 dias em 2017, 4 dias em 2014 e 7 dias pré-oficiais em 2024). Isso dá 5 284 linhas oficiais (7 070 com 2024) nos 1 790 pontos do ML.
2. **Quais variáveis entraram?** As 22 dos grupos TS, CORE, VERTICAL, CURRENT e DYNAMICS, todas com dado. As de 50 m têm 35 % de NaN, onde o fundo é mais raso.
3. **O baseline mudou ao receber CROCO?** Sim: com CORE, a PR-AUC conjunta foi de 0,426 para 0,483. Mas dentro do mês melhora em 1 de 3 meses e piora em 2. As probabilidades mudam pouco (média 0,016).
4. **Correntes ajudaram?** Não: −0,055, pior que o embaralhado.
5. **Vorticidade ajudou?** Não: associação |ρ| ≤ 0,05, e o grupo DYNAMICS piorou (−0,061).
6. **Transporte ajudou?** Não: associação fraca (|ρ| ≈ 0,1) e dentro do grupo que piorou.
7. **MLD e N² ajudaram?** É o melhor candidato: MLD tem associação −0,26 e o CORE ganhou +0,057, mas sem estabilidade entre meses.
8. **Alguma variável é promissora?** **temperature_surface, HBL e MLD do CROCO**, com a ressalva do mês de 2014.
9. **Venceu o controle embaralhado?** CORE e "todas", sim (0,483 e 0,467 contra ~0,43). CURRENT e DYNAMICS, não.
10. **Justifica gerar mais anos de CROCO?** **Não.** No máximo, alguns meses-alvo (setembro/outubro) para confirmar o CORE.
11. **Próximo experimento mínimo:** ~6 meses completos de set/out de anos diferentes, com spin-up, rodados em segmentos ≤ 1,5 h (~60 h no total). Depois, repetir este piloto só com o CORE.

> **Este é um teste piloto exploratório. A série CROCO disponível é insuficiente para concluir que as features físicas melhoram a capacidade de previsão futura do modelo.**
