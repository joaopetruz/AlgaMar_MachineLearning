# Etapa 13.5 — Integração CROCO/Navier–Stokes → ML

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage`. Nada foi enviado ao GitHub. A física não foi refeita e nenhum modelo com CROCO foi treinado.

**Código:**
- `algamar_ml/croco_ml.py`: grupos, carga validada, cubo e proveniência;
- `config/ml.toml`: flags `use_croco_*`;
- `algamar_ml/pipeline.py` e `features/horizonte.py`: as flags agora valem;
- `algamar_ml/croco.py`: o modo NOWCAST passa a ser bloqueado;
- `comparar_com_fisica.py --grupos` e `treinar_previsao.py`: excluem os pontos NAO_USAR;
- `etapa13_05_verifica_croco.py`: verificação com dado real;
- `tests/test_croco_ml.py`: 11 testes; a suíte tem **90**.

**Saídas:** `resultados/etapa13_05/verificacao_croco.json` e `proveniencia_features_croco.json`.

---

## 1. Baseline isolado

Com todas as flags `use_croco_* = false` (o padrão), o modelo é exatamente o baseline.

**Verificação:** refiz a previsão operacional de set/2026 com o `baseline-hist-api-v1` pelo pipeline novo. As 21 480 probabilidades saíram **idênticas, bit a bit**, às da Etapa 13.4.

O E0 sazonal (`croco.py`) também continua igual, porque o CROCO só entra por `juntar_croco.py`.

## 2. Flags (config/ml.toml [features])

```
use_baseline_features = true    # padrão histórico do mês-alvo (= use_historical_features)
use_api_features      = true    # estado atual de clorofila e SST em t0
use_croco_ts          = false
use_croco_core        = false
use_croco_vertical    = false
use_croco_current     = false
use_croco_dynamics    = false
caminhos.croco_mensal = "dados/croco/croco_mensal.parquet"   # tabela mensal do contrato 13.0
```

## 3. Grupos

Os grupos usam o contrato 13.0. Nenhum status foi reclassificado.

| grupo | features (colunas `<f>_mean` no arquivo) | status (contrato 13.0) |
|---|---|---|
| **CROCO_TS** | temperature_surface, salinity_surface | VALIDADA |
| **CROCO_CORE** | TS + mld, n2_max | VALIDADA / DERIVADA_VALIDADA (N²) |
| **CROCO_VERTICAL** | temperature_20m, salinity_20m, temperature_50m, salinity_50m, temperature_bottom, salinity_bottom | VALIDADA (NaN onde h < profundidade) |
| **CROCO_CURRENT** | u_east_surface, v_north_surface, speed_of_mean_current | NAO_VALIDADA_OBS |
| **CROCO_DYNAMICS** | ssh, w_10m, w_50m, relative_vorticity_surface, normalized_vorticity_surface, horizontal_divergence_surface, hbl, transport_east, transport_north | NAO_VALIDADA_OBS; o W leva também NAO_INTERPRETAR_COMO_UPWELLING_AUTOMATICAMENTE |

**Justificativas:**
- **5 m e 10 m ficam fora do VERTICAL:** são redundantes com a superfície (Spearman 0,89–0,99 na amostra 13.0, 13.2A).
- **A direção fica fora:** é uma variável circular; use u e v.
- **`speed_of_mean_current` é o módulo da corrente média do mês:** é o análogo da antiga `corrente_velocidade` do ML.
- **CROCO_CORE contém o TS:** ligar os dois não duplica colunas.
- **Registro:** os grupos também estão em `contrato_croco/feature_sets.json`, em `grupos_13_5`.

## 4. Join

- **Chave:** (latitude, longitude, ano, mes), com `round(5)` nas coordenadas e `ano` e `mes` inteiros, as mesmas regras da 13.0/13.2A.
- **Chave repetida após o arredondamento é recusada.** A tabela do CROCO é única por chave.
- **Alinhamento com o cubo:** por ponto da grade, com `validate="many_to_one"` (cada linha CROCO casa com um único ponto).
- **No E0 sazonal:** `junta_croco` usa `validate="one_to_one"`.

**Verificação com dado real** (o mensal do piloto 13.1, nov–dez/2017, 3 706 linhas):
- os 1 790 pontos do ML têm CROCO: **0 sem CROCO**;
- 63 pontos do CROCO estão fora do dataset do ML (pontos da grade CHL que o ML não tem): ignorados;
- 3 580 linhas casadas no cubo, 0 duplicatas.

## 5. Validade espacial

- **NAO_USAR_ESTUARIO_CANAL:** 4 dos pontos do ML, de 12 na grade toda. Ficam NaN em **todos os meses** e, no treino ou comparação com CROCO, saem do dataset (`excluir_nao_usar`), com o **mesmo conjunto de linhas no baseline**.
- **NAO_VALIDADO_COSTEIRO:** fica, com a flag (`validade_por_ponto`), para estratificar as métricas.
- **Valor de validade desconhecido:** é recusado.

## 6. NaN físico

Nada é imputado.
- O carregamento **recusa** o arquivo se houver valor a 20 ou 50 m (ou W a 10 ou 50 m) onde `h_d2_m` é menor que a profundidade.
- No piloto real, T e S a 50 m são NaN em 1 304 linhas e a 20 m em 324, todas com h menor.
- Na comparação com física, a linha **não** é descartada por NaN físico; só o ponto NAO_USAR e o mês sem CROCO saem.

## 7. Alinhamento temporal (por horizonte)

**A física entra sempre como o estado do mês de emissão t0** (`croco_<f>_t0`).
- Para o alvo τ = t0 + h, isso é uma defasagem de **h meses**, coerente com cada horizonte.
- **Nunca** há valor do mês-alvo nem de meses posteriores a t0.

**Testes:**
- o valor é o mesmo em todos os h da mesma emissão;
- mudar o CROCO depois de t0 não muda nenhuma feature;
- mês sem CROCO vira NaN.

No piloto real, para cada (ponto, t0) a física é única nos 12 horizontes.

**No E0 sazonal,** que prevê o mês t com dados até t−1:
- só `FORECAST_1M` (lag 1) e `FORECAST_SAZONAL` (lag 12) são aceitos;
- **`NOWCAST` agora levanta erro**, a não ser com `permite_nowcast=True`, que é só para diagnóstico.

## 8. Proveniência

Cada feature física registra (exemplo de `croco_mld_t0`, em `proveniencia_features_croco.json`):
- source = CROCO;
- coluna `mld_mean` do contrato;
- feature_version 13.0.0;
- unit m;
- status_validation VALIDADA;
- timestamp: média do mês civil UTC, no mês de emissão t0;
- período: 2017-11 a 2017-12;
- grid/version: `build_d2_algamar_final_v1`, commit CROCO ab63b583, SHA do executável e SHA da grade `e7346752…`, lidos do `provenance.json` da produção.

Quando o Parquet traz metadados por coluna (contrato 13.0), a **unidade é conferida** com o contrato. Uma divergência, como K em vez de degC, recusa o arquivo.

## 9. Testes

| tema | teste |
|---|---|
| schema | coluna do grupo ou `validade_espacial` faltando → erro |
| join e duplicatas | chave repetida após `round(5)` → erro; 1:1 com o cubo |
| NaN esperados | 50 m em h < 50 m continua NaN; valor "imputado" ali → erro |
| unidades | unidade no metadado diferente do contrato → erro; faixa física |
| coordenadas | pontos do ML sem CROCO e pontos CROCO fora do ML contados |
| cobertura temporal | mês faltando listado; vira NaN sem preenchimento |
| origem | proveniência completa por feature |
| lag temporal | física = mês t0 em todos os h; mudar o CROCO depois de t0 não altera features |
| validade espacial | NAO_USAR todo NaN; valor de validade desconhecido → erro |
| baseline | sem flags CROCO, a lista de features é a do baseline e a previsão é idêntica |

## 10. Respostas

1. **O ML está pronto para receber CROCO?** **Sim, a parte de software está pronta.** Basta colocar a tabela mensal do contrato 13.0 em `dados/croco/croco_mensal.parquet` e ligar as flags. A verificação com o mensal real do piloto passou de ponta a ponta.
2. **Quais features entram em cada grupo?** As da tabela da §3.
3. **O join está correto?** Sim:
   - chave (lat, lon, ano, mes) com `round(5)` e chave única;
   - casamento 1:1 com os pontos;
   - 0 pontos do ML sem CROCO no piloto;
   - NAO_USAR excluído.
4. **Existem riscos de leakage?** Pela construção, não: a física só entra no mês t0, e os testes provam isso em todos os h. Ficam três riscos para vigiar:
   - **NOWCAST no E0 sazonal**, agora bloqueado por padrão;
   - **meses com produção incompleta** (`mes_completo = False`, como os dois meses do piloto): a média mensal de um mês parcial ainda é passado em relação a t0+h, mas é menos representativa. Ficam com a flag;
   - **reanálise nas forçantes:** GLORYS e ERA5 assimilam dados próximos de t0. Para o hindcast isso é aceitável, mas operar exige forçantes NRT (13.4).
5. **O baseline continua funcionando sem CROCO?** Sim: a previsão é bit a bit idêntica à da 13.4 com as flags desligadas, e o E0 sazonal não mudou.
6. **Está tudo pronto para começar os experimentos?** **O encanamento sim; os dados não.**
   - Hoje o CROCO cobre só **2 meses** (piloto nov–dez/2017). O treino (alvos até 2018) e a validação (2019–2021) precisam de **anos** de CROCO para a comparação baseline × baseline + core × baseline + dynamics fazer sentido.
   - Faltam:
     - a produção CROCO plurianual (13.1: cerca de 5 dias de máquina por ano; 25 anos ≈ 4 meses);
     - a sua escolha do modelo de previsão oficial (pendente da 13.3);
     - um controle GLORYS opcional, para separar ganho do CROCO de ganho de "mais física".
