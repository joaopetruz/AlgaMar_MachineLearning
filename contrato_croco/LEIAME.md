# Entrada das features do CROCO (Navier-Stokes) no ML

As features físicas vêm do `cfd_navier_stokes` (CROCO D2, 1,5 km), no contrato 13.0. Os arquivos desta pasta são cópias de lá:

| arquivo | origem |
|---|---|
| `feature_contract.csv`, `dataset_schema.json`, `units.csv`, `provenance_schema.json` | `outputs/etapa13_00_interface_ml/` |
| `feature_sets.json`, `join_spec.json` | `outputs/etapa13_02a_auditoria_ml/` |

## Como usar

1. Coloque o Parquet mensal do CROCO em `dados/croco/`. Ele precisa ter:
   - uma linha por (latitude, longitude, ano, mes);
   - as colunas `<feature>_mean`;
   - a flag `validade_espacial`.
2. Junte com o dataset do ML:
   `python juntar_croco.py dados/croco/<arquivo>.parquet --conjunto CROCO_CORE --modo FORECAST_SAZONAL`
3. Treine e compare com o E0:
   `python treinar_modelo_sazonal.py --entrada dados/dados_features_sazonal_sp_croco.csv --croco CROCO_CORE --modo FORECAST_SAZONAL`

## Regras (aplicadas pelo `algamar_ml/croco.py`)

- **Chave:** `round(5)` nas coordenadas, com join `one_to_one`. Chave repetida é recusada.
- **Pontos `NAO_USAR_ESTUARIO_CANAL`:** saem em todos os meses. Os pontos `NAO_VALIDADO_COSTEIRO` ficam, com a flag.
- **Modos temporais:**
  - `NOWCAST` usa o mês t;
  - `FORECAST_SAZONAL` usa `__lag12`, por data;
  - `FORECAST_1M` usa `__lag1`.

  Não misture modos numa mesma métrica.
- **Valores abaixo do fundo:** ficam NaN e nunca são imputados. O XGBoost trata NaN nativamente.
- **Mesmas linhas para todos:** para comparar com o E0, os dois modelos precisam usar as mesmas linhas, ou seja, o período coberto pelo CROCO.
