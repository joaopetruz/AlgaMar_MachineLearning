"""Etapa 13.6 — estratificação por horizonte (h = 1..12): o ganho da física sobrevive além do mês seguinte?

Esquema de previsão da 13.3 (E0_forecast, um modelo por horizonte, features do E0 reancoradas em t0, mesmo split e
hiperparâmetros, limiar por horizonte na validação). Variantes (física e satélite SEMPRE no mês de emissão t0):
  F0  E0_forecast
  F1  F0 + SST/SSS de satélite em t0
  F2  F0 + GLORYS T/S de superfície + MLD em t0
  F3  F1 + GLORYS MLD em t0           (o que a física acrescenta além do satélite)
Mesmas linhas (sem os pontos NAO_USAR_ESTUARIO_CANAL), seed 42.
Saída: resultados/etapa13_06/glorys/por_horizonte.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from algamar_ml import config, croco_ml
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import E0_FORECAST_FEATURES, constroi_linhas, cubo_do_painel, indice_mes
from algamar_ml.features.sazonal import ANO_FIM_HOTSPOT, hotspots_do_treino
from algamar_ml.metricas import escolhe_limiar
from algamar_ml.modelos.treino import divide

t00 = time.time()
cfg = config.carrega()
O = Path("resultados/etapa13_06/glorys"); O.mkdir(parents=True, exist_ok=True)
mensal = pd.read_csv(cfg["caminhos"]["historico_equipe"])
cubo = cubo_do_painel(mensal)
hs = hotspots_do_treino(mensal, ano_fim=ANO_FIM_HOTSPOT)

# GLORYS mensal → matrizes [mês, ponto] no cubo
ds = xr.open_dataset("dados/brutos/glorys_mensal/glorys_mensal_2000_2024.nc").isel(depth=0)
la, lo = xr.DataArray(cubo.pontos.latitude.to_numpy(), dims="p"), xr.DataArray(cubo.pontos.longitude.to_numpy(), dims="p")
it = ds.interp(latitude=la, longitude=lo)
ti = np.array([indice_mes(t.year, t.month) for t in pd.to_datetime(it.time.values)]) - cubo.t_inicio
fis = {}
for nome, var in (("glorys_ts_t0", "thetao"), ("glorys_ss_t0", "so"), ("glorys_mld_t0", "mlotst")):
    a = np.full((cubo.T, len(cubo.pontos)), np.nan, dtype=np.float32)
    ok = (ti >= 0) & (ti < cubo.T)
    a[ti[ok]] = it[var].values[ok]
    fis[nome] = a

linhas = constroi_linhas(cubo, cfg["alvo"]["limiar"], range(int(indice_mes(2001, 1)), cubo.t_fim + 1), range(1, 13),
                         fisica=fis, hotspots=hs)
meta = pd.read_parquet("../../cfd_navier_stokes/outputs/etapa13_01_piloto_producao/features_mensais/features_mensais_auditoria.parquet")
meta = meta[["latitude", "longitude", "validade_espacial"]].drop_duplicates(["latitude", "longitude"])
nao_usar = set(map(tuple, meta.loc[meta.validade_espacial == croco_ml.NAO_USAR, ["latitude", "longitude"]].round(5).to_numpy()))
pid_nao = [p for p, a, b in zip(cubo.pontos.point_id, cubo.pontos.latitude.round(5), cubo.pontos.longitude.round(5)) if (a, b) in nao_usar]
linhas = linhas[linhas["floracao"].notna() & ~linhas["point_id"].isin(pid_nao)].reset_index(drop=True)
tr, va, te = divide(linhas, cfg["split"])
y = linhas["floracao"].astype(int).to_numpy()
h = linhas["h"].to_numpy()
print(f"linhas {len(linhas):,} (sem {len(pid_nao)} pontos NAO_USAR) | {time.time()-t00:.0f}s", flush=True)

SAT = ["sst_t0", "sss_t0"]
VAR = {"F0_E0_forecast": E0_FORECAST_FEATURES, "F1_F0+satelite_TS_t0": E0_FORECAST_FEATURES + SAT,
       "F2_F0+GLORYS_TS+MLD_t0": E0_FORECAST_FEATURES + ["glorys_ts_t0", "glorys_ss_t0", "glorys_mld_t0"],
       "F3_F1+GLORYS_MLD_t0": E0_FORECAST_FEATURES + SAT + ["glorys_mld_t0"]}
P = dict(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss", tree_method="hist")
prob = {}
for nome, feats in VAR.items():
    p = np.full(len(linhas), np.nan)
    for k in range(1, 13):
        mk = tr & (h == k)
        w = float((y[mk] == 0).sum() / (y[mk] == 1).sum())
        m = XGBClassifier(scale_pos_weight=w, **P).fit(linhas.loc[mk, feats], y[mk])
        s = (va | te) & (h == k)
        p[s] = m.predict_proba(linhas.loc[s, feats])[:, 1]
    prob[nome] = p
    print(nome, f"{time.time()-t00:.0f}s", flush=True)

reg = regiao(linhas["longitude"], cfg["regioes"])
blocos = (linhas["tau"] // 12).astype(str).to_numpy() + "|" + reg.astype(str) + "|" + ((linhas["tau"] % 12) // 3).astype(str).to_numpy()
rng = np.random.default_rng(0)
lin = []
for k in range(1, 13):
    st = te & (h == k)
    yy, bb = y[st], blocos[st]
    ub = np.unique(bb); ix = {b: np.where(bb == b)[0] for b in ub}
    am = [np.concatenate([ix[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(300)]
    ref = prob["F0_E0_forecast"][st]
    for nome in VAR:
        pp = prob[nome][st]
        d = [average_precision_score(yy[a], pp[a]) - average_precision_score(yy[a], ref[a]) for a in am]
        lin.append({"h": k, "modelo": nome, "n": int(st.sum()), "pr_auc": average_precision_score(yy, pp),
                    "delta_vs_F0": average_precision_score(yy, pp) - average_precision_score(yy, ref),
                    "ic95_lo": float(np.percentile(d, 2.5)), "ic95_hi": float(np.percentile(d, 97.5))})
t = pd.DataFrame(lin)
t.to_csv(O / "por_horizonte.csv", index=False, float_format="%.5f")
print(t.pivot(index="h", columns="modelo", values="delta_vs_F0").round(3).to_string(), flush=True)
print(t.pivot(index="h", columns="modelo", values="pr_auc").round(3).to_string(), flush=True)
print("ok", round(time.time() - t00), "s")
