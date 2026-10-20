"""Etapa 13.5 — verificação com dado REAL: a tabela mensal do piloto CROCO 13.1 (nov–dez/2017) entra no ML.

1. carrega e valida os 5 grupos (schema, chave 1:1, unidades, faixa, NaN físico, cobertura, proveniência);
2. alinha à grade do ML (1 790 pontos) e monta as linhas (ponto, t0, h) com a física no mês de emissão;
3. confere que, com todas as flags CROCO desligadas, o baseline gera exatamente a mesma previsão de antes.
Saída: resultados/etapa13_05/verificacao_croco.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml import config, croco_ml, pipeline
from algamar_ml.features.horizonte import constroi_linhas, cubo_do_painel, indice_mes
from algamar_ml.modelos import treino as tr
from algamar_ml.previsao import preve

CFD = Path("../../cfd_navier_stokes/outputs/etapa13_01_piloto_producao")
ARQ = CFD / "features_mensais" / "features_mensais_auditoria.parquet"
O = Path("resultados/etapa13_05"); O.mkdir(parents=True, exist_ok=True)
cfg = config.carrega()
prov = json.loads((CFD / "provenance.json").read_text(encoding="utf-8"))
todas = [f for g in croco_ml.GRUPOS.values() for f in g if f not in []]
todas = list(dict.fromkeys(todas))
tabela, rel = croco_ml.carrega(ARQ, todas, proveniencia=prov)

mensal = pd.read_csv(cfg["caminhos"]["historico_equipe"])
cubo = cubo_do_painel(mensal)
_, rel_pts = croco_ml.carrega(tabela, todas, pontos_ml=cubo.pontos)
mats, rel_cubo = croco_ml.para_cubo(tabela, cubo, todas)
lin = constroi_linhas(cubo, cfg["alvo"]["limiar"], [indice_mes(2017, 11), indice_mes(2017, 12)], range(1, 13), fisica=mats)
val = croco_ml.validade_por_ponto(tabela, cubo)
nao_usar = np.where(val == croco_ml.NAO_USAR)[0]
cols = [c for c in lin.columns if c.startswith("croco_")]
nao_nan = {c: int(lin[c].notna().sum()) for c in cols}
check_lag = bool((lin.groupby(["point_id", "t0"])[cols].nunique(dropna=False) <= 1).all().all())   # mesmo valor em todo h

# baseline sem CROCO: mesma previsão de antes (operacional_2026-09, etapa 13.4)
modelo, cartao = tr.carrega(Path(cfg["caminhos"]["modelos"]) / "baseline-hist-api-v1")
cfg_b = config.carrega()
cfg_b["features"].update(cartao["flags"])
assert not croco_ml.grupos_ativos(cfg_b["features"])
cubo_p, painel = pipeline.carrega_cubo(cfg_b)
mats_b, nomes_b = pipeline.fisica(cfg_b, cubo_p)
nova = preve(modelo, cartao, cubo_p, cfg_b, fisica=mats_b, t0=indice_mes(2026, 9))
antiga = pd.read_parquet("previsoes/operacional_2026-09/previsao_pontos.parquet")
igual = bool(np.array_equal(nova["bloom_probability"].to_numpy(), antiga["bloom_probability"].to_numpy()))

res = {
    "arquivo_croco": str(ARQ), "linhas": rel["linhas"], "cobertura": rel["cobertura"], "avisos": rel["avisos"],
    "validade_pontos": rel["validade"], "pontos_ml_sem_croco": rel_pts["pontos_ml_sem_croco"],
    "pontos_croco_fora_do_ml": rel_pts["pontos_croco_fora_do_ml"],
    "nan_por_feature": rel["nan_por_feature"], "proveniencia_exemplo": {k: rel["proveniencia"][k] for k in list(rel["proveniencia"])[:2]},
    "cubo": rel_cubo, "pontos_NAO_USAR_no_cubo": int(len(nao_usar)),
    "linhas_previsao_2017_11_12": int(len(lin)), "valores_croco_nao_nan": nao_nan,
    "fisica_igual_em_todos_os_h_da_mesma_emissao": check_lag,
    "nao_usar_todo_nan": bool(lin[lin.point_id.isin(nao_usar)][cols].isna().all().all()),
    "baseline_sem_croco": {"features_fisicas": nomes_b, "previsao_identica_a_13_4": igual, "features": len(cartao["features"])},
}
(O / "verificacao_croco.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
(O / "proveniencia_features_croco.json").write_text(json.dumps(rel["proveniencia"], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("proveniencia_exemplo",)}, ensure_ascii=False, indent=1, default=str))
