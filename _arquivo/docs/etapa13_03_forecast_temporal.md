# Etapa 13.3 — Previsão temporal explícita (1 a 12 meses)

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage` (base a629b2c). Nada foi enviado ao GitHub e o CROCO não entrou.

**Saídas:**
- `resultados/etapa13_03/`: `horizonte_e0.json`, `metricas_por_horizonte_teste.csv`, `previsao_2025_x_observado.csv` e `resultados_e0_forecast.json` (manifesto, cartão e métricas de validação e teste por horizonte);
- `modelos/e0-forecast-v1/`: 12 modelos e o cartão;
- `previsoes/e0-forecast-v1_2024-12/`: previsão real emitida em dez/2024 para jan–dez/2025.

**Scripts:** `etapa13_03_horizonte_e0.py` e `etapa13_03_e0_forecast.py`.

---

## 1. O que o modelo realmente previa

A prova é empírica (`horizonte_e0.json`). Para linhas reais (ponto, mês T), alterei os dados de **um único mês** T−k (k = 0…13) naquele ponto e verifiquei quais features da linha T mudavam.

| modelo | menor k que altera features | classe | features afetadas pelo próprio mês T |
|---|---|---|---|
| **E0_original** (repositório da equipe) | **k = 0** | **NOWCAST**: usa o próprio mês-alvo | `clorofila_media_3anos`, `salinidade_media_3anos`, `clorofila_media_historica_mes`, `salinidade_media_historica_mes` |
| **E0_corrigido** (13.2B) | **k = 1** | **FORECAST_1M no máximo**: precisa do mês T−1 | nenhuma |

Em ambos, os 4 "ano anterior" usam T−12, e as médias de 36 meses usam T−1 a T−36.

**Respostas:**
- O E0_original não era uma previsão futura (opção A: o próprio mês). Pelo código: `rolling(36)` incluindo t e climatologia de todos os anos.
- O E0_corrigido serve no máximo para o mês seguinte (opção B). Usado com 2 ou mais meses de antecedência, ele exigiria dados de meses que ainda não aconteceram.
- Nenhum dos dois prevê "o mesmo mês do próximo ano" (opção C). Só as features de ano anterior permitiriam isso.
- O site (`index.html`) desenhava uma curva sazonal com números fixos digitados, como mostrou a auditoria de 08/10.

## 2. Nowcast × forecast

| tipo | definição | onde está |
|---|---|---|
| NOWCAST | estado do mês atual, usando dados do próprio mês | E0_original e `/predictions` antigo. Não é usado para previsão. |
| FORECAST_1M | mês seguinte, com dados até t0 | E0_corrigido; E0_forecast h = 1 |
| FORECAST_NM | N meses à frente, com dados até t0 | **E0_forecast h = 1…12** (esta etapa) e `baseline-hist-api-v1` (08/10) |

## 3. Alvo temporal e horizontes

Cada linha é **(ponto, mês de emissão t0, horizonte h)**, com h = 1…12. As features usam **só meses ≤ t0**, e o alvo é a floração (`clorofila_maxima > 1,5`) no mês **t0 + h**.

Exemplo: dados até jan/2024 (t0) → h = 1 prevê fev/2024 … h = 12 prevê jan/2025.

O código está em `algamar_ml/features/horizonte.py` (`constroi_linhas`, `E0_FORECAST_FEATURES`). As 12 features conceituais do E0 foram **reancoradas** em t0:

| feature E0 | no forecast (alvo τ = t0 + h) | mês mais recente usado |
|---|---|---|
| mês, lat, lon | mês-alvo, lat, lon | — |
| dist_hotspot (H2) | igual (km, hotspots do treino) | — |
| 4 × "ano anterior" | o mesmo mês-alvo no ano anterior (τ − 12) | τ − 12 ≤ t0 (por isso h ≤ 12) |
| climatologias de clorofila e salinidade | do mês-alvo, só nos anos anteriores a τ | ≤ τ − 12 ≤ t0 |
| médias de 36 meses | terminando em **t0** | t0 |
| (novo) h | horizonte em meses | — |

**Teste de equivalência:** em h = 1 (emissão T−1), as features são **idênticas** às do E0_corrigido da linha T (`test_e0_forecast_h1_igual_ao_e0_corrigido`).

## 4. Arquitetura (escolhida só na validação)

| arquitetura | descrição | uso |
|---|---|---|
| **A** direta multi-horizonte, um modelo | h como feature, todos os horizontes juntos | testada |
| **B** direta, um modelo por horizonte | 12 modelos com as mesmas features | testada |
| recursiva | prevê t0+1 e realimenta | **descartada**: exigiria prever também clorofila, SST e SSS do mês seguinte, e o erro se acumula |
| multi-output | um modelo com 12 saídas | **descartada**: equivale a A ou B no XGBoost e complica a validação por horizonte |

A regra foi fixada antes de rodar: vence a maior PR-AUC média nos 12 horizontes da **validação** 2019–2021.
- A: 0,715
- **B: 0,721 → escolhida.**

O teste não participou da escolha. Com B, cada horizonte tem seu `scale_pos_weight` e seu limiar.

**Hiperparâmetros:** os do E0 (300 árvores, profundidade 6, lr 0,1, seed 42), sem tuning.

**Split** (por emissão e alvo, sem nenhum alvo em dois conjuntos):

| conjunto | regra | linhas |
|---|---|---|
| treino | alvo ≤ 2018-12 | 4 348 860 |
| validação | emissão ≥ 2019-01 e alvo ≤ 2021-12 | 631 860 |
| teste | emissão ≥ 2022-01, alvo ≤ 2024-12 | 633 660 |

**Limiar:** máximo F1 na validação, **por horizonte**, congelado antes do teste.

## 5. Métricas por horizonte (teste 2022–2024)

`metricas_por_horizonte_teste.csv`:

| h | PR-AUC | climatologia | ROC-AUC | F1 | recall | precisão | Brier | log loss | limiar |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,746 | 0,768 | 0,917 | 0,697 | 0,761 | 0,643 | 0,115 | 0,398 | 0,68 |
| 2 | 0,755 | 0,776 | 0,913 | 0,694 | 0,743 | 0,650 | 0,124 | 0,421 | 0,75 |
| 3 | 0,742 | 0,772 | 0,912 | 0,693 | 0,752 | 0,643 | 0,128 | 0,454 | 0,75 |
| 4 | 0,751 | 0,766 | 0,912 | 0,693 | 0,766 | 0,633 | 0,131 | 0,455 | 0,74 |
| 5 | 0,743 | 0,760 | 0,907 | 0,677 | 0,747 | 0,619 | 0,138 | 0,468 | 0,75 |
| 6 | 0,730 | 0,751 | 0,901 | 0,672 | 0,721 | 0,629 | 0,140 | 0,480 | 0,78 |
| 7 | 0,742 | 0,751 | 0,906 | 0,682 | 0,735 | 0,636 | 0,141 | 0,481 | 0,79 |
| 8 | 0,729 | 0,744 | 0,901 | 0,665 | 0,774 | 0,583 | 0,144 | 0,493 | 0,70 |
| 9 | 0,765 | 0,740 | 0,909 | 0,682 | 0,773 | 0,610 | 0,130 | 0,431 | 0,69 |
| 10 | 0,760 | 0,744 | 0,910 | 0,687 | 0,695 | 0,679 | 0,129 | 0,437 | 0,83 |
| 11 | 0,767 | 0,746 | 0,913 | 0,689 | 0,669 | 0,710 | 0,133 | 0,450 | 0,88 |
| 12 | 0,768 | 0,746 | 0,911 | 0,700 | 0,734 | 0,669 | 0,133 | 0,448 | 0,80 |

A climatologia é a frequência histórica de floração no ponto e mês-alvo, só com anos anteriores.

**Leitura honesta:**
- O E0_forecast **funciona em todos os 12 horizontes**: PR-AUC de 0,73 a 0,77 e ROC-AUC de 0,90 a 0,92, sem degradar com h.
- **Mas ele fica abaixo da climatologia em h = 1…8** (−0,01 a −0,03) e só a supera um pouco em h = 9…12.
- **Com as features do E0, o modelo ainda não acrescenta informação além do padrão histórico de cada ponto e mês.**
- O modelo `baseline-hist-api-v1` (08/10) usa o estado de t0 e anomalias, e parte da climatologia. Ele fica +0,005 a +0,009 acima da climatologia em todos os horizontes, num teste mais longo (2022–2026/09). É o mesmo achado: hoje quase todo o sinal é climatológico.

## 6. Previsão real (teste mínimo do pipeline)

- **Emissão:** dez/2024, o último mês do dataset congelado.
- **Saída:** 21 480 linhas (1 790 pontos × 12 meses) em `previsoes/e0-forecast-v1_2024-12/previsao_pontos.parquet/csv`, com `resumo_regioes.csv`.
- **Colunas:**
  - `prediction_date`, `issue_month`, `target_date`, `forecast_horizon_months`;
  - `point_id`, `latitude`, `longitude`, `region`;
  - `bloom_probability`, `predicted_class`, `risk_level` (BAIXO/MODERADO/ALTO), `decision_threshold`;
  - `climatological_probability`, `model_version`.
- **Campos não preenchidos:** município, espécie e toxicidade, que não existem nos dados.
- **Região:** os 3 setores de longitude do `config/ml.toml`, ainda provisórios (DÚVIDA 2).

**Conferência com o que aconteceu em 2025.** O modelo nunca viu 2025. A clorofila observada vem do painel MY (`previsao_2025_x_observado.csv`):

| alvo | jan | fev | mar | abr | mai | jun | jul | ago | set | out | nov | dez |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| PR-AUC modelo | 0,88 | 0,91 | 0,85 | 0,92 | 0,92 | 0,68 | 0,89 | 0,93 | 0,83 | 0,89 | 0,76 | 0,63 |
| PR-AUC climatologia | 0,77 | 0,89 | 0,88 | 0,93 | 0,90 | 0,90 | 0,91 | 0,94 | 0,94 | 0,93 | 0,72 | 0,64 |

O pipeline previu um ano real fora da amostra. O resultado confirma o quadro do teste: o modelo bate a climatologia em alguns meses e perde em outros.

## 7. Como o leakage temporal é evitado

1. As features são matrizes [mês, ponto]. Cada linha lê **só o índice t0** (estado) ou **τ − 12, τ − 24, …** (padrão do mês-alvo, todos ≤ t0 porque h ≤ 12).
2. A climatologia e as médias são expansivas ou móveis só para trás; não há estatística calculada sobre o dataset inteiro.
3. Os hotspots (H2) e o `scale_pos_weight` vêm só do treino.
4. O split é por emissão e alvo, sem nenhum alvo em dois conjuntos. Limiar e arquitetura vêm da validação.
5. **Testes, 52 passando:**
   - mudar todos os meses depois de t0 não muda nenhuma feature da emissão t0, **em cada h = 1…12** (`test_e0_forecast_sem_dado_depois_de_t0_em_todos_os_horizontes` e o teste geral da 08/10);
   - h > 12 é recusado;
   - a virada de ano está correta;
   - mês faltante vira NaN.

## 8. Localização

Cada linha carrega `point_id`, `latitude` e `longitude` (grade CHL L4 4 km, 1 790 pontos oceânicos, WGS84), e a saída acrescenta `region`. As previsões podem ser agregadas por região (`resumo_regioes.csv`) ou mapeadas ponto a ponto.

## 9. Respostas

1. **O ML fazia previsão futura real?** Não. O E0_original era um nowcast, porque usava o próprio mês-alvo. O E0_corrigido chega a no máximo 1 mês à frente.
2. **Qual era o horizonte real?** 0 meses no original; 1 mês no corrigido, quando emitido no fim do mês anterior.
3. **Como funciona a nova arquitetura?** É previsão direta: para cada emissão t0 e horizonte h, as features vêm só até t0 e o alvo é o mês t0 + h. Foram 12 modelos (arquitetura B), escolhida na validação, com as features do E0 reancoradas.
4. **Como h = 1–12 foi implementado?** Pelo `constroi_linhas(..., horizontes=1..12)`: cada horizonte gera as linhas (ponto, t0, h), e o "ano anterior" do mês-alvo (τ − 12) limita o horizonte a 12.
5. **Como o leakage é evitado?** Veja o §7. A garantia vem de testes que alteram o futuro e conferem que nada muda.
6. **Como cada previsão é ligada à posição?** Por `point_id`, lat/lon da grade CHL e região.
7. **Quais horizontes funcionam?** Todos, de h = 1 a 12, tanto no pipeline quanto nas métricas e na previsão real.
8. **O que ainda não funciona?** Nenhum horizonte supera a climatologia de forma consistente com as features do E0, e as regiões são provisórias. A operação em tempo real com o E0 também não funciona: a SSS recente (NRT) é inconsistente, e o E0 depende de salinidade.
9. **O que falta para a operação real?**
   - escolher o modelo de previsão oficial (pergunta abaixo);
   - atualizar os dados mensalmente (o `atualizar_dados_atuais.py` já existe para CHL e SST);
   - decidir o tratamento da SSS;
   - definir as regiões;
   - ligar a API `/forecast` ao modelo escolhido;
   - só então, as features físicas do CROCO.

## 10. Decisão pendente (pergunta ao Tulio)

**Qual modelo de previsão passa a ser a referência oficial antes do CROCO?**

- **A. `baseline-hist-api-v1` (recomendado).** Já fica ≥ climatologia em todos os horizontes, usa o estado de t0 e funciona com os dados de 2026.
- **B. `e0-forecast-v1`.** É fiel às features do E0, mas fica abaixo da climatologia em h = 1…8.
- **C.** E0_forecast com a base climatológica da opção A. Exige um novo treino, e a escolha seria feita na validação.

**Recomendação: A**, mantendo o `e0-forecast-v1` e a climatologia como referências obrigatórias em toda comparação futura.
