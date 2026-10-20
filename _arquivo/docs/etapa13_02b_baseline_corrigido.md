# Etapa 13.2B — Baseline do ML corrigido (E0_original × E0_corrigido)

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage` (base ae7371d; nada enviado ao GitHub).

**Resultado: A.** O baseline está corrigido e validado. O hotspot H2 foi escolhido pelo Tulio em 09/10 e congelado como padrão em `constroi_features`.

**O que não foi feito:**
- nenhuma feature do CROCO no modelo;
- nenhum tuning de hiperparâmetros;
- alvo e dataset não foram alterados;
- o teste não foi usado para nenhuma escolha.

**Saídas** em `resultados/etapa13_02b/`:
- `hotspots_validacao.json`;
- `H2/`, com:
  - `metricas_E0_original.json`, `metricas_E0_corrigido.json` e `metricas_diagnostico_features_antigas_protocolo_novo.json`;
  - `config_modelos.json`, `limiar.json` e `manifest.json`;
  - `comparacao_teste_2022_2024.csv`;
- `testes.txt` e `arquivos_modificados.txt`.

**Scripts:** `etapa13_02b_baseline.py --hotspot H2` e `etapa13_02b_hotspots.py`.

---

## 1. Baseline original preservado (E0_original)

`algamar_ml.features.constroi_features_original` reproduz exatamente o `feature_engineering_sazonal.py` de cc3145c. A 13.2A mostrou que ele reproduz o CSV `fb7e1579…` com diferença máxima de 7e-15.

O protocolo também é o original:
- treino ≤ 2021 e teste 2022–2024;
- sem validação;
- limiar fixo 0,5;
- XGBoost 300/6/0,1, `scale_pos_weight` do treino, seed 42.

Nada da lógica original foi apagado.

## 2. Baseline corrigido (E0_corrigido)

Mantém o mesmo algoritmo, os mesmos hiperparâmetros, o mesmo alvo (`clorofila_maxima > 1,5`) e as mesmas 12 features conceituais. As correções estão em `algamar_ml/features/sazonal.py` (commit 5cf5fa1, de 08/10) e nesta etapa:

| problema | original | corrigido |
|---|---|---|
| climatologia por (ponto, mês) | média de 2000–2024, incluindo o ano t e os anos de teste | média só dos anos **anteriores** ao da linha (`climatologia_passada`) |
| média de 36 meses | `rolling(36)` por linha, **incluindo t** | meses de calendário **t−36 a t−1**, mínimo de 12 válidos (`media_movel_passada`) |
| ano anterior | `shift(12)` por posição (pega o mês errado em 900 linhas) | junção **por data**: 2020-05 busca 2019-05; se não existir, NaN (`lag_por_data`) |
| hotspots | 2 pontos fixos, escolhidos no mapa de 2024 (ano de teste); distância em graus | **H2**: recalculados só com os anos ≤ 2018, distância em km (§3) |
| split | treino ≤ 2021 e teste > 2021 | treino 2001–2018, validação 2019–2021, teste 2022–2024 |
| limiar | 0,5 fixo; os scripts olhavam o teste | máximo F1 na validação (grade de 0,05 a 0,95, passo 0,01), congelado antes do teste |

As linhas são as mesmas nas duas versões: 501 870 linhas comuns. As 900 linhas com "ano anterior" errado saem da original.

## 3. Hotspots (decisão: H2)

As alternativas foram medidas **só na validação** (`hotspots_validacao.json`):

| opção | PR-AUC | ROC-AUC | F1 | log loss |
|---|---|---|---|---|
| H0: atuais, em graus | 0,754 | 0,893 | 0,667 | 0,399 |
| H1: atuais, em km | 0,762 | 0,895 | 0,673 | 0,389 |
| **H2: recalculados com 2000–2018, em km** | 0,760 | 0,895 | 0,670 | 0,387 |
| H3: sem dist_hotspot | 0,752 | 0,892 | 0,666 | 0,405 |

**Regra do H2:**
- pega os 2 pontos de maior clorofila média nos anos ≤ 2018, separados por pelo menos 50 km;
- o resultado é (−24,19; −46,77) e (−23,85; −46,06), a 32 e 55 km dos atuais.

Os hotspots são um parâmetro fixo, ajustado no treino como um scaler. Nenhum dado de validação ou teste entra.

**Escolhido: H2.** Mantém o conceito, elimina a origem no ano de teste e tem desempenho equivalente na validação. `constroi_features` agora usa H2 por padrão (`ANO_FIM_HOTSPOT = 2018`); `constroi_features_original` continua com os hotspots e a distância originais.

## 4. Comparação no teste 2022–2024 (64 440 linhas, 13 446 positivos)

| modelo | PR-AUC | ROC-AUC | F1 | recall | precisão | Brier | log loss | limiar |
|---|---|---|---|---|---|---|---|---|
| **E0_original** | **0,800** | 0,911 | 0,702 | 0,733 | 0,673 | 0,094 | 0,323 | 0,50 |
| features antigas no protocolo novo (diagnóstico) | 0,778 | 0,902 | 0,686 | 0,697 | 0,675 | 0,113 | 0,387 | 0,71 |
| **E0_corrigido (H2)** | **0,737** | 0,899 | 0,689 | 0,700 | 0,678 | 0,116 | 0,419 | 0,73 |

**Matrizes de confusão** (TN, FP, FN, TP):
- E0_original: 46 213, 4 781, 3 590, 9 856;
- E0_corrigido: 46 515, 4 479, 4 033, 9 413.

**Leitura:**
- **PR-AUC caiu 0,063 (−7,9 %).** Desse total:
  - 0,022 vêm do protocolo: o treino é 3 anos menor e o limiar não é mais escolhido de forma otimista;
  - 0,041 vêm das features sem vazamento.
- **O F1 no limiar ficou parecido:** 0,689 contra 0,702.
- **A calibração piorou um pouco:** Brier 0,094 → 0,116 e log loss 0,323 → 0,419. Parte disso vem do treino mais curto e de o limiar sair da validação.
- **Os números do E0_original estavam inflados:** o vazamento inflava sobretudo o ranking (PR-AUC).

## 5. Reprodutibilidade (`manifest.json`, `config_modelos.json`)

| item | valor |
|---|---|
| Python | 3.13.2 |
| pandas | 2.3.3 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| scipy | 1.18.1 |
| seed | 42 |
| hiperparâmetros | n_estimators 300, max_depth 6, learning_rate 0,1, scale_pos_weight = negativos/positivos do treino, eval_metric logloss |
| dataset mensal | `dados_mensais_2000_2024_sp_com_salinidade.csv`, SHA256 `f58dcfa8…` (o congelado na 13.2A) |
| features (conteúdo) | E0_original `635d8e85…`; E0_corrigido (H2) `2fbdb176…` |
| commit base | ae7371d + as mudanças desta etapa (`arquivos_modificados.txt`) |
| ambiente | `C:\Users\Tulio\.venvs\algamar-ml` |

O `requirements.txt` ainda usa faixas de versão. As versões exatas usadas estão no manifesto.

## 6. Testes

**50/50 passam** (`testes.txt`). Os específicos desta correção:

| garantia | teste |
|---|---|
| climatologia sem dados futuros | `test_climatologia_so_usa_anos_anteriores`; `test_features_nao_enxergam_o_futuro` (mudar 2004+ não altera nenhuma feature de antes) |
| média de 36 meses só com passado | `test_media_3anos_exclui_o_mes_corrente`; `test_features_do_mes_t_nao_dependem_da_clorofila_de_t` |
| lag de 12 meses por data | `test_lag_por_data_respeita_lacuna` (mês faltante vira NaN, nunca outro mês) |
| separação treino/validação/teste | `test_split_disjunto_e_por_ano` |
| limiar nunca escolhido no teste | `test_limiar_escolhido_so_na_validacao` (espia a função: uma chamada, só com os rótulos da validação); `test_limiar_nao_muda_quando_o_teste_muda` |
| hotspots sem dados futuros | `test_hotspots_do_treino_ignoram_anos_futuros`; `test_constroi_features_usa_hotspots_so_do_treino`; `test_distancia_km` |

## 7. Arquivos modificados

- `algamar_ml/avaliacao.py`: log loss.
- `algamar_ml/features/sazonal.py`: `distancia_hotspot_km`, `hotspots_do_treino` e H2 como padrão em `constroi_features`.
- Novos: `etapa13_02b_baseline.py`, `etapa13_02b_hotspots.py`, `tests/test_baseline_13_2b.py`, `docs/etapa13_02b_baseline_corrigido.md` e `resultados/etapa13_02b/`.
- As correções de 08/10 já estavam no commit 5cf5fa1: features sem vazamento, split e limiar na validação.

## 8. Pendências

1. **Formulário do site.** O `index.html` ainda pede um `dist_hotspot` digitado à mão, com os hotspots antigos e em graus. Ele precisa acompanhar o H2 quando o produto for refeito.
2. **Modelo do .pkl da API.** O `.pkl` em uso é um RandomForest, não o XGBoost avaliado (revisão, item 7). O E0 oficial deve ser o XGBoost avaliado aqui.
3. **Salinidade preenchida.** A salinidade costeira preenchida do ramo `girino` não entra neste dataset congelado. Fica para decidir.

## 9. Respostas

1. **Leakage corrigido?** Sim: a climatologia usa só os anos anteriores, a média de 3 anos exclui t, o limiar sai da validação e os hotspots saem só do treino.
2. **Shift de 12 meses?** Sim, agora por data, com NaN quando o mês falta.
3. **Média de 36 meses só com passado?** Sim, de t−36 a t−1.
4. **Threshold da validação?** Sim: 0,73 (máximo F1 em 2019–2021), congelado antes do teste.
5. **Quanto mudou?** No teste 2022–24:
   - PR-AUC 0,800 → 0,737 (−0,063);
   - ROC-AUC 0,911 → 0,899;
   - F1 0,702 → 0,689;
   - Brier 0,094 → 0,116.
6. **Pronto como baseline oficial?** Sim. O E0_corrigido (H2) é o baseline oficial a partir desta etapa.
7. **Próxima etapa:** congelar o E0_corrigido como baseline oficial e então testar baseline, baseline + core físico e baseline + dinâmica, com séries CROCO longas o suficiente.
