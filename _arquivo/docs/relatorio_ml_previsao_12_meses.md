# ML do AlgaMar: relatório da fase "previsão de 12 meses + preparação para o Navier–Stokes"

**Data:** 08/10/2026. **Ramo local:** `algamar/correcao-leakage` (a partir do `girino` 83d5f2b). Nada foi enviado ao GitHub.

**Legenda:** **IMPLEMENTADO** · **PREPARADO** (estrutura pronta e testada, falta o dado real) · **EM DESENVOLVIMENTO** · **NÃO IMPLEMENTADO**.

---

## Resumo

| objetivo | situação |
|---|---|
| previsão temporal explícita, h = 1…12 meses, a partir do último mês observado | **IMPLEMENTADO** |
| dados atuais das APIs (Copernicus) entrando na previsão | **IMPLEMENTADO** para clorofila e SST, até set/2026; SSS desligada (viés do produto NRT) |
| histórico usado para aprender relações t → t+h, sem vazamento | **IMPLEMENTADO** e testado |
| onde: probabilidade por ponto de 4 km e por região | **IMPLEMENTADO**; as regiões são provisórias (DÚVIDA 2) |
| saída padrão (data de emissão, data-alvo, horizonte, lat/lon, região, probabilidade, risco, versão) | **IMPLEMENTADO** |
| métricas por horizonte, região, estação e calibração; referências climatologia e persistência | **IMPLEMENTADO** |
| baseline salvo (histórico + APIs) | **IMPLEMENTADO**: `baselines/baseline-hist-api-v1/` |
| camada Navier–Stokes (fontes → derivadas → contrato → alinhamento → feature store → ML) | **PREPARADO**; testada com a amostra real do CROCO (nov/2017) e dados sintéticos |
| comparação sem física × com física (bootstrap, controle embaralhado) | **PREPARADO**; o encanamento foi testado com ruído sintético |
| treinamento final | **NÃO FEITO, de propósito**: será após a integração do Navier–Stokes |

**Resultado principal, sem maquiagem:**
- O sistema prevê os 12 meses, mas **com as variáveis atuais (clorofila e SST) o modelo quase não supera a climatologia**: PR-AUC de 0,783 contra 0,778, ΔPR-AUC = +0,005 com IC 95 % de −0,005 a +0,019.
- Ou seja, hoje o "quando e onde" vem sobretudo do padrão histórico de cada ponto e mês, e o estado atual do mar acrescenta pouco.
- É exatamente essa lacuna que as features físicas do Navier–Stokes vão ser testadas para preencher.

---

## ETAPA 1: Inspeção do projeto

**Encontrado:**
- 28 scripts soltos e um modelo de um mês só, sem horizonte;
- uma API que lê um CSV estático até dez/2024;
- uma página que repete números digitados;
- 95 arquivos de dados da equipe, de 2000 a 2024.

**APIs disponíveis** (consulta ao catálogo da Copernicus, 08/10/2026):

| variável | produto MY/REP: última data | produto NRT: período | resolução | unidade |
|---|---|---|---|---|
| clorofila (CHL L4 gap-free) | **30/09/2026** | 30/09–07/10/2026 | 4 km, diária | mg m⁻³ |
| SST (OSTIA) | 31/03/2026 | 17/01/2024–07/10/2026 | 0,05°, diária | K → °C |
| SSS (multi-obs) | 15/12/2024 | 01/01/2024–02/10/2026 | 0,125°, diária | PSU |
| vento (L4 NRT, horário) | — | 13/06/2024–07/10/2026 | 0,125° | m/s (não usado; DÚVIDA 3) |

O login da Copernicus já estava configurado pelo Tulio nesta máquina. As credenciais não foram lidas.

- **Problemas:** os listados na auditoria anterior (`../auditoria_previsao_temporal_2026-10-08.md`).
- **Próxima etapa:** diagnóstico e dúvidas.

## ETAPA 2: Diagnóstico e dúvidas

Enviei 4 "DÚVIDAS NECESSÁRIAS" na conversa. **Ainda sem resposta.** Segui com a recomendação de cada uma, e todas são trocáveis em `config/ml.toml`:

1. **Alvo:** mantido clorofila > 1,5 mg/m³.

   **Achado novo que reforça a dúvida:**
   - 82 dos 1 790 pontos (a faixa costeira) passam de 1,5 em ≥ 90 % de todos os meses;
   - 390 pontos quase nunca passam (≤ 5 %).

   Na costa, a previsão fica em ~99 % todos os meses, o que **não informa nada**. Um limiar relativo por ponto (opção B) resolveria isso.
2. **Regiões:** 3 setores de longitude (sul, centro e norte).
3. **Variáveis do baseline:** clorofila, SST e SSS. **A SSS acabou desligada**, porque o produto NRT é inconsistente (Etapa 7).
4. **SSS costeira:** sem preenchimento. NaN é aceito nativamente pelo modelo.

## ETAPA 3: Correção dos dados

**Alterado:** `processamento/mensal.py` refaz o processamento da equipe (interpolação bilinear para a grade CHL e média e máximo do mês civil), e `processamento/painel.py` monta o painel com proveniência por variável.

**Testes:**
- o reprocessamento de **2024** a partir dos NetCDF da equipe é **idêntico** ao CSV da equipe: diferença ≤ 1e-6 na clorofila (float32) e ≤ 3e-14 em T e S;
- a clorofila MY baixada hoje para dez/2024 é **idêntica, bit a bit**, à da equipe. É a mesma versão do produto.

**Resultado:**
- o painel `dados/processados/painel_mensal.parquet` tem 1 790 pontos, de **jan/2000 a set/2026**, com 561 015 linhas e 0 chaves duplicadas;
- `painel_mensal.manifesto.json` registra os SHA256, a lista de meses e os NaN.

## ETAPA 4: Correção do pipeline

**Alterado:**
- **`algamar_ml/features/horizonte.py`:** cada linha é (ponto, mês de emissão t0, horizonte h). As features usam **só meses ≤ t0**, e o alvo é a floração em t0+h.
- **Grupos de features (por flag):**
  - **base:** h, mês de emissão, mês-alvo, lat, lon;
  - **api:** estado em t0, t0−1 e t0−2, médias de 3 e 36 meses e anomalias contra a climatologia dos anos anteriores;
  - **histórico:** climatologia do mês-alvo, frequência histórica de floração, número de anos e o mesmo mês do ano anterior (t0+h−12 ≤ t0);
  - **física:** estado físico em t0.
- **Sem estatística global:** toda climatologia é expansiva, só com anos anteriores.

**Testes:**
- mudar todos os dados depois de t0* não muda nenhuma feature com t0 ≤ t0*, enquanto o alvo muda;
- a virada de ano está correta;
- mês faltando vira NaN, e não outro mês;
- a climatologia além do fim do painel é igual à calculada dentro dele;
- h > 12 é recusado.

## ETAPA 5: Previsão temporal

**IMPLEMENTADA:** um modelo direto multi-horizonte (XGBoost), com h como feature.

- **Ponto de partida climatológico:** o modelo parte da climatologia do ponto e mês (log-odds, suavizada para a taxa geral quando há poucos anos) e aprende o desvio explicado pelo estado atual.
- **Por que essa escolha:** sem essa base, o primeiro treino ficou **abaixo** da climatologia (PR-AUC 0,751 contra 0,778). Com ela, o modelo fica igual ou um pouco acima, e a calibração melhora (Brier 0,097 → 0,083).
- **Pesos:** sem peso de classe, porque a saída é usada como probabilidade.

## ETAPA 6: Previsão espacial

**IMPLEMENTADA** por ponto da grade de 4 km (1 790 pontos) e por região. `espacial.py` tem:
- distância de haversine;
- vizinho mais próximo com distância máxima;
- interpolação bilinear;
- atribuição de região.

**Não disponível nos dados:** município, estação de coleta, espécie e toxicidade. Esses campos não são inventados.

## ETAPA 7: APIs

**Alterado:** `apis/copernicus.py` (catálogo MY/REP + NRT; para cada mês, usa o MY se ele cobre o mês inteiro, senão o NRT) e `atualizar_dados_atuais.py`. Foram baixados 87 arquivos:
- 2025-01 a 2026-09 de CHL, SST e SSS;
- 2024 de SST e SSS NRT, para checar a consistência.

**Consistência NRT × MY em 2024** (`resultados/consistencia_nrt_2024.json`):

| variável | viés NRT − MY | RMSE | correlação | decisão |
|---|---|---|---|---|
| SST | +0,07 °C | 0,17 °C | 0,998 | usada (NRT só de abr a set/2026) |
| SSS | **+0,60 PSU** (até +1,87 em out) | 0,84 | **0,61** | **desligada** (`use_salinity = false`); religar quando houver SSS consistente |

**Resultado:** o mês de emissão é **set/2026**, com CHL MY e SST NRT.

## ETAPA 8: Validação temporal

**Split**, por mês de emissão e mês-alvo, sem nenhum alvo em dois conjuntos:

| conjunto | período dos alvos | linhas | uso |
|---|---|---|---|
| treino | 2001-02 a 2018-12 | 4 348 860 | ajuste do modelo |
| validação | 2019-02 a 2021-12 | 631 860 | limiar (máximo F1) e early stopping |
| teste | 2022-02 a 2026-09 | 1 084 740 | só relatório |

**Métricas no teste** (`baselines/baseline-hist-api-v1/cartao.json`):

| | PR-AUC | ROC-AUC | Brier | F1 | recall | precisão |
|---|---|---|---|---|---|---|
| **modelo base (histórico + APIs)** | **0,783** | **0,905** | 0,083 | 0,670 | 0,584 | 0,786 |
| climatologia (frequência histórica do ponto e mês) | 0,778 | 0,904 | 0,082 | — | — | — |
| persistência (mesmo mês do ano anterior) | 0,389 | 0,725 | 0,179 | — | — | — |

- **Limiar:** 0,39, escolhido na validação. A taxa de floração no teste é de 19,5 %.
- **Modelo × climatologia:** ΔPR-AUC = +0,005, com IC 95 % de −0,005 a +0,019. O bootstrap usou 15 blocos de ano × região (`resultados/baseline_vs_climatologia.json`). **A diferença não é significativa.**
- **Por horizonte:** o PR-AUC vai de 0,792 a 0,795 em h = 1–3 e cai para cerca de 0,777 em h = 9–12. Ele fica acima da climatologia em todos os horizontes, mas por pouco.
- **Por região** (PR-AUC modelo / climatologia):
  - sul: 0,888 / 0,893;
  - centro: 0,806 / 0,805;
  - norte: 0,664 / 0,643.
- **Por estação:** DJF 0,823 / 0,806; MAM 0,893 / 0,885; JJA 0,753 / 0,749; SON 0,725 / 0,733.
- **Calibração:** o modelo subestima um pouco nas faixas de 0,1 a 0,4. Por exemplo, previu 0,24 e o observado foi 0,33. O teste teve mais floração que o treino.
- **Early stopping:** parou na árvore 46, mais um sinal de que há pouco a aprender além da climatologia com estas variáveis.

## ETAPA 9: Previsão de 12 meses

**IMPLEMENTADA:** `prever_12_meses.py` → `previsoes/2026-09/`. Os arquivos gerados são:
- `previsao_pontos.parquet/csv`, com 21 480 linhas = 1 790 pontos × 12 meses;
- `resumo_regioes.csv`;
- `pico_por_ponto.csv`;
- `metadados.json`.

Cópia do resumo em `resultados/previsao_2026-09/`.

**Probabilidade média por região, out/2026 → set/2027:**

| região | out | nov | dez | jan | fev | mar | abr | mai | jun | jul | ago | set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sul | 0,17 | 0,19 | 0,14 | 0,20 | 0,27 | 0,25 | 0,22 | 0,23 | 0,30 | **0,47** | **0,48** | 0,30 |
| centro | 0,14 | 0,17 | 0,10 | 0,14 | 0,19 | 0,18 | 0,15 | 0,17 | 0,24 | 0,27 | **0,32** | 0,22 |
| norte | 0,09 | 0,10 | 0,07 | 0,09 | 0,11 | 0,10 | 0,09 | 0,12 | 0,19 | 0,20 | **0,23** | 0,15 |

O maior risco previsto é em jul–ago/2027, sobretudo no setor sul.

**Ressalva:** como o modelo ≈ climatologia, isso é essencialmente o padrão sazonal histórico, levemente ajustado pelo estado de set/2026.

## ETAPA 10: Testes

São **44 testes, todos passando** (`python -m pytest`). Eles cobrem:
- datas e horizontes;
- ausência de vazamento;
- schema da saída e do contrato;
- coordenadas, haversine e regiões;
- escolha do produto MY/NRT;
- processamento mensal (sintético e real de 2024);
- painel real;
- split sem sobreposição;
- salvar e carregar modelo;
- endpoints `/forecast` (12 meses com ano, 404 fora da área, 503 sem previsão);
- camada física (derivadas num campo de rotação sólida, mês UTC e cobertura, contrato, feature store, física entrando em t0 e não no mês-alvo);
- bootstrap.

## ETAPA 11: Arquitetura

A organização está em `LEIAME_ML.md`.

- **Pacote `algamar_ml/`:** `apis`, `processamento`, `features`, `modelos`, `metricas`, `previsao`, `espacial`, `fisica`, `api_previsao`.
- **Configuração:** `config/ml.toml`, com as feature flags.
- **Contratos:** `contrato/`.
- **API:** `/forecast`, `/forecast/regioes`, `/forecast/mapa` e `/forecast/meta`; `previsao_12_meses.html` é a página nova.
- **O que continua funcionando:**
  - `/predict`, `/predictions`, `/marine-data` e `/climatologia`; agora são opcionais e marcados como "classificação de mês observado";
  - `index.html`, com um aviso apontando para a página nova;
  - os scripts sazonais antigos.

## ETAPA 12: Preparação para o Navier–Stokes

**PREPARADO.** O ML nunca lê o solver diretamente:

```
solver → fisica/fontes.py → derivadas.py → contrato.py (padroniza e valida) → alinhamento.py (mês UTC, pontos ML) → feature_store.py → ML
```

**Adaptadores:**
- `FonteTabelaContrato13`, para o export do CROCO D2 do `cfd_navier_stokes`;
- `FonteGradeUV`, para qualquer campo u/v em grade lat/lon (ex.: solver NS 2D).

Trocar o solver = escrever outro adaptador. A física entra como **estado em t0** (`ns_<feature>_t0`), nunca como valor do mês-alvo.

**Teste com dado real:** a amostra 13.0 do CROCO (nov/2017) entrou na store com os 1 790 pontos casados, 9 features e o manifesto de proveniência.

## ETAPA 13: Interface e contrato das features físicas

**`contrato/features_fisicas.csv`** tem 20 features:
- `u`, `v`, `velocidade` e `direcao_sin`/`direcao_cos`;
- `energia_cinetica`, `vorticidade`, `divergencia` e `okubo_weiss`;
- `pressao` (SSH), transportes, `gradiente_temperatura` e `w_10m`;
- T e S de superfície e fundo, `mld` e `n2_max`.

Para cada uma o contrato traz tipo, unidade, origem, descrição, frequência, resolução, obrigatoriedade, faixa permitida, tratamento de ausentes, status de validação e conjuntos (`NS_CORE`, `NS_CIRCULACAO` e `NS_FULL`).

**`contrato/features_ml.csv`** documenta todas as features do modelo, com origem e momento: t0, alvo(≤ t0) ou fixo.

**Alinhamento:**
- **Temporal:** mês civil UTC. Recusa tempo sem fuso e registros duplicados. Meses com menos de 80 % dos dias viram NaN.
- **Espacial:** vizinho mais próximo por haversine, até `dist_max_km` (padrão de 3 km), com a distância registrada; bilinear disponível.

**Comparação:** `comparar_com_fisica.py` usa as mesmas linhas, o mesmo split e as mesmas seeds para os dois modelos. Ele calcula o ΔPR-AUC com IC por bootstrap por horizonte e um controle com a física embaralhada.

**Teste de encanamento:** rodei com **ruído sintético** no lugar da física. Ele deu ΔPR-AUC = −0,002 (IC −0,006 a +0,003), ou seja, **nenhum ganho**, como deveria. Isso valida o método e **não é um resultado científico**.

## ETAPA 14: Validação final

**Executados:** os 44 testes; o painel real; o treino da base; a previsão real de out/2026 a set/2027; a API com os 7 locais da página nova (todos a ≤ 9,2 km de um ponto da grade); e a comparação com ruído sintético.

---

## O que falta (próxima fase)

1. **Respostas às 4 dúvidas**, principalmente a 1: limiar fixo × relativo por ponto.
2. **Produção CROCO de 2000 a 2026** (13.2A) exportada no contrato 13.0, e depois `python ingerir_fisica.py tabela <export>`.
3. **`python comparar_com_fisica.py --seeds 42,43,44 --controle-embaralhado`.** A física só é considerada útil se o IC 95 % do ΔPR-AUC ficar acima de 0 na maioria dos horizontes e o controle não mostrar ganho.
4. **Será necessário retreinar o modelo após a integração dos novos dados e das features do Navier-Stokes.** O `baseline-hist-api-v1` é base de arquitetura, não o modelo final.
5. **SSS:** encontrar uma salinidade consistente para os meses recentes, ou usar a do CROCO.
6. **Opcional:** vento (NRT disponível) como variável de API, se a DÚVIDA 3 for B.

## Arquivos alterados ou criados nesta fase

| tipo | arquivos |
|---|---|
| novos (código) | `algamar_ml/{config, espacial, metricas, pipeline, previsao, api_previsao}.py`, `algamar_ml/apis/`, `algamar_ml/processamento/`, `algamar_ml/features/{horizonte, contrato_ml}.py`, `algamar_ml/modelos/`, `algamar_ml/fisica/` |
| novos (configuração e contratos) | `config/ml.toml`, `contrato/` |
| novos (scripts) | `atualizar_dados_atuais.py`, `construir_painel.py`, `treinar_previsao.py`, `prever_12_meses.py`, `ingerir_fisica.py`, `comparar_com_fisica.py`, `checar_consistencia_nrt.py` |
| novos (página e docs) | `previsao_12_meses.html`, `LEIAME_ML.md`, `docs/` |
| novos (testes) | `tests/test_{horizonte, fisica, previsao, api_previsao, integracao_dados}.py` |
| alterados | `api.py` (router `/forecast`; o modelo antigo ficou opcional), `index.html` (aviso), `.gitignore` |
| movido | `algamar_ml/features.py` → `algamar_ml/features/sazonal.py`, com reexport para compatibilidade |
| baseline | `baselines/baseline-hist-api-v1/` (modelo JSON de 648 KB + cartão) |
