"""Litoral de SP inteiro (2026-10-09): a versão treinada com a área ampliada (risco-sat-clim-sp-v1) perde algo nos pontos
antigos, e como ela se sai nos pontos novos (Iguape, Ilha Comprida, Cananéia)?

Modelos lidos do registro (SHA256 conferido; nada é retreinado): baseline-hist-api-v1 (oficial), risco-sat-clim-v1
(enxuta, área da equipe) e risco-sat-clim-sp-v1 (enxuta, área SP total). Mesmo teste do config (emissões ≥ 2022-01,
h = 1..12), painel SP total. Os pontos antigos têm dados idênticos aos do CSV da equipe (conferencia_equipe.json).
- pontos antigos: Δ PR-AUC contra o oficial;
- pontos novos: Δ PR-AUC contra a climatologia (os modelos antigos nunca viram esses pontos; ficam só como referência).
IC 95 % por bootstrap em blocos ano do alvo × região × trimestre (200 reamostragens).
Saída: resultados/sp_total/{comparacao.csv, por_regiao.csv, resultados.json}
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from algamar_ml import config, pipeline, registro
from algamar_ml.espacial import regiao
from algamar_ml.modelos import treino as tr

t00 = time.time()
O = Path("resultados/sp_total"); O.mkdir(parents=True, exist_ok=True)
cfg = config.carrega()
VERSOES = {"oficial": "baseline-hist-api-v1", "enxuta": "risco-sat-clim-v1", "sp_total": "risco-sat-clim-sp-v1"}
cubo, _ = pipeline.carrega_cubo(cfg)
linhas = pipeline.dataset(cfg, cubo)
_, _, te = tr.divide(linhas, cfg["split"])
y = linhas["floracao"].to_numpy(dtype=int)
e = cfg["area"]["caixa_equipe"]
lat, lon = linhas["latitude"].to_numpy(), linhas["longitude"].to_numpy()
antigo = (lon >= e[0]) & (lon <= e[1]) & (lat >= e[2]) & (lat <= e[3])
print(f"linhas {len(linhas):,} | teste {te.sum():,} | antigos {(te & antigo).sum():,} | novos {(te & ~antigo).sum():,}", flush=True)

prob = {}
for nome, versao in VERSOES.items():
    m, cartao, _ = registro.carrega(cfg["caminhos"]["modelos"], versao)
    p = np.full(len(linhas), np.nan); p[te] = tr.probabilidade(m, linhas[te], cartao); prob[nome] = p
prob["climatologia"] = linhas["freq_floracao_alvo"].to_numpy(dtype=float)

reg = np.asarray(regiao(linhas["longitude"], cfg["regioes"])).astype(str)
blocos = (linhas["tau"] // 12).astype(str).to_numpy() + "|" + reg + "|" + ((linhas["tau"] % 12) // 3).astype(str).to_numpy()
rng = np.random.default_rng(0)


def avalia(mask, rot, ref_nome, nomes):
    ok = mask & ~np.isnan(prob["climatologia"])
    yy = y[ok]; bb = blocos[ok]; ub = np.unique(bb); ix = {b: np.where(bb == b)[0] for b in ub}
    am = [np.concatenate([ix[b] for b in rng.choice(ub, len(ub))]) for _ in range(200)]
    ref = prob[ref_nome][ok]; out = []
    for nome in nomes:
        pp = prob[nome][ok]
        d = [average_precision_score(yy[a], pp[a]) - average_precision_score(yy[a], ref[a]) for a in am]
        out.append({**rot, "modelo": nome, "n": int(ok.sum()), "pontos": int(len(np.unique(linhas.loc[ok, "point_id"]))),
                    "taxa_floracao": float(yy.mean()), "pr_auc": average_precision_score(yy, pp), "roc_auc": roc_auc_score(yy, pp),
                    "brier": brier_score_loss(yy, pp), "referencia": ref_nome,
                    "delta_pr_auc": average_precision_score(yy, pp) - average_precision_score(yy, ref),
                    "ic95_lo": float(np.percentile(d, 2.5)), "ic95_hi": float(np.percentile(d, 97.5))})
    return out


todos = list(prob)
geral = pd.DataFrame(avalia(te & antigo, {"conjunto": "pontos antigos (área da equipe)"}, "oficial", todos)
                     + avalia(te & ~antigo, {"conjunto": "pontos novos (litoral sul)"}, "climatologia", todos)
                     + avalia(te, {"conjunto": "litoral SP inteiro"}, "climatologia", ["sp_total", "climatologia"]))
por_reg = pd.DataFrame([r for nome in np.unique(reg) for r in avalia(te & (reg == nome), {"regiao": nome}, "climatologia",
                                                                       ["sp_total", "climatologia"])])
geral.to_csv(O / "comparacao.csv", index=False, float_format="%.5f"); por_reg.to_csv(O / "por_regiao.csv", index=False, float_format="%.5f")
pts = linhas[te].drop_duplicates("point_id")
sem_sst = int(pd.read_parquet(cfg["caminhos"]["painel"]).groupby(["latitude", "longitude"]).temperatura_media
                         .apply(lambda s: s.isna().all()).sum())
json.dump({"versoes": VERSOES, "pontos_teste": int(len(pts)), "pontos_antigos": int(len(np.unique(linhas.loc[te & antigo, "point_id"]))),
           "pontos_sem_sst_nunca": sem_sst, "segundos": round(time.time() - t00)},
          open(O / "resultados.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 200)
print(geral.drop(columns=["referencia"]).round(4).to_string())
print(por_reg.round(4).to_string())
print("pontos sem SST em nenhum mês:", sem_sst)
print("ok", round(time.time() - t00), "s")
