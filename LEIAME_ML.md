# Mapa de Risco de Floração AlgaMar

O que o sistema faz: com o **histórico** (2000–2024) e o **estado ambiental atual** de satélite baixado da Copernicus (clorofila e temperatura da superfície), ele estima, para cada ponto da grade de 4 km do litoral de SP e para cada um dos **12 meses seguintes ao último mês observado**, a probabilidade de floração (clorofila máxima mensal > 1,5 mg/m³), com nível de risco e região.

> **Status (fechamento, 2026-10-09):** HINDCAST VALIDADO. Na prática, o produto é **risco climatológico ajustado pelo satélite**: acerta onde e em que mês costuma florescer (PR-AUC ~0,78), mas não supera a climatologia em anomalias. A física Navier–Stokes (CROCO) e a reanálise GLORYS foram testadas e **não melhoraram** a previsão; os grupos sem sinal ficam bloqueados em `config/ml.toml [features].sem_sinal` e o código continua no repositório como histórico. Detalhes e defesa do resultado: `docs/RELATORIO_FECHAMENTO_RISCO_FLORACAO.md`. Uso atual: `python algamar.py previsao` (ver `docs/etapa13_08_pipeline_final.md`).

> **Organização (2026-10-19):** scripts de experimento, exploração antiga, resultados, relatórios de etapa e saídas antigas foram movidos para `_arquivo/` (ver `_arquivo/LEIAME.md`). Na raiz ficou só o que gera a previsão final, o treino e o modelo sazonal antigo da equipe.

> As seções seguintes descrevem a fase de 08/10, quando a camada física ainda era a aposta.

## Fluxo

```
Copernicus (CHL, SST, SSS; MY/REP + NRT)      histórico da equipe 2000–2024
        │ atualizar_dados_atuais.py                    │
        ▼                                              ▼
processamento/mensal.py ── grade CHL 4 km, mês civil ──► processamento/painel.py ──► painel_mensal.parquet (+ manifesto)
                                                                                      │
Navier–Stokes / CROCO ─► fisica/fontes → derivadas → contrato → alinhamento → feature_store (Parquet + manifesto)
                                                                                      │ (flag use_navier_stokes_features)
                                                                                      ▼
                         features/horizonte.py: linhas (ponto, t0, h=1..12), features só até t0
                                                                                      ▼
                         modelos/treino.py: split temporal, XGBoost com base climatológica, limiar na validação
                                                                                      ▼
                         previsao.py ─► previsoes/<t0>/ (pontos, regiões, pico) ─► API /forecast ─► previsao_12_meses.html
```

## Comandos (venv: `C:\Users\Tulio\.venvs\algamar-ml`)

| passo | comando |
|---|---|
| baixar os meses novos (usa o login Copernicus já configurado) | `python atualizar_dados_atuais.py` |
| montar o painel mensal | `python construir_painel.py` |
| treinar e registrar uma versão nova | `python algamar.py treinar --versao NOVA` (`--ensemble` para o ensemble de ordens) |
| gerar a previsão de 12 meses (modelo oficial do `config/ml.toml`) | `python algamar.py previsao` (`--baixar` atualiza os dados antes) → `saidas/<run_id>/` |
| subir a API | `uvicorn api:app` → `/forecast`, `/forecast/regioes`, `/forecast/mapa`, `/forecast/meta` e `/coastal-risks` (formato do dashboard; o backend repassa em `/api/ml/coastal-risks`). **Testar o ML:** abrir `http://127.0.0.1:8000/testar` (previsão feita num mês do passado × o que aconteceu; `/teste/hindcast?emissao=AAAA-MM`). Serve a execução de `saidas/ULTIMA.json`; só sem ela cai em `previsoes/` (legado de `prever_12_meses.py`, hoje em `_arquivo/`) |
| dashboard completo no seu PC (sem o backend da outra máquina) | `python dashboard_local.py` → API local em `http://localhost:3000/api` (mesma porta do backend; qualquer e-mail e senha entram; `/api/ml/coastal-risks` vem direto do ML). Abrir: `http://localhost:3000/software/pages/login.html`, ou pelo Live Server (5501) com a linha `http://localhost:3000/api` ativa no `software/js/api-config.js` do frontend. Frontend padrão: `OneDrive/Desktop/Algamar_Page_Software` (`--frontend PASTA` para outro) |
| ver no navegador | `python -m http.server 5599` e abrir `http://127.0.0.1:5599/previsao_12_meses.html` (API em `127.0.0.1:8000`) |
| testes | `python -m pytest` |

## Organização

| pasta/arquivo | responsabilidade |
|---|---|
| `config/ml.toml` | caminhos, área, alvo, horizontes, split, hiperparâmetros, **feature flags**, regiões |
| `algamar_ml/apis/copernicus.py` | catálogo dos produtos (MY/REP e NRT), escolha por mês, download |
| `algamar_ml/processamento/` | diários → mensais na grade CHL (mesmo método da equipe; 2024 reproduzido) e painel com proveniência |
| `algamar_ml/features/horizonte.py` | dataset multi-horizonte sem vazamento |
| `algamar_ml/features/sazonal.py` | features do modelo antigo (compatibilidade) |
| `algamar_ml/modelos/treino.py` | split temporal, treino, cartão do modelo |
| `algamar_ml/metricas.py` | PR-AUC, ROC-AUC, Brier, F1, calibração, por horizonte/região/estação, bootstrap |
| `algamar_ml/previsao.py` | previsão de 12 meses no formato padrão + resumos por região e pico |
| `algamar_ml/espacial.py` | haversine, regiões, vizinho mais próximo, bilinear |
| `algamar_ml/fisica/` | camada Navier–Stokes: fontes, derivadas, contrato, alinhamento, feature store |
| `contrato/features_fisicas.csv` | contrato das features físicas |
| `contrato/features_ml.csv` | contrato de todas as features do modelo |
| `api.py` + `algamar_ml/api_previsao.py` | API (endpoints antigos preservados + `/forecast`) |
| `previsao_12_meses.html` | página da previsão de 12 meses |

Scripts antigos (`feature_engineering_sazonal.py`, `treinar_modelo_sazonal.py`, `salvar_modelo_sazonal.py`, `index.html`, `/predict`, `/predictions`) continuam funcionando. Eles classificam um mês e **não** fazem previsão futura.

## Formato da previsão (`previsoes/<t0>/previsao_pontos.parquet`)

São 12 linhas por ponto:

| campo | descrição |
|---|---|
| `prediction_date` | data em que a previsão foi gerada |
| `issue_month` | último mês observado (t0) |
| `target_date` | mês previsto |
| `forecast_horizon` | h, de 1 a 12 |
| `point_id`, `latitude`, `longitude` | ponto da grade |
| `region` | região |
| `bloom_probability` | probabilidade de floração prevista |
| `climatological_probability` | frequência histórica de floração no ponto e mês |
| `risk_level` | BAIXO / MODERADO / ALTO |
| `model_version` | versão do modelo |
| `feature_flags` | grupos de features usados |
| `data_sources` | produto de cada variável no mês t0 |

**Não preenchidos, por falta de dado:** município, estação de coleta, espécie e toxicidade.
