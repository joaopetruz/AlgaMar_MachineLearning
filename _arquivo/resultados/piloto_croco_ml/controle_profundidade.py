"""Controle extra do piloto: o ganho de CROCO_TS/CORE é só geografia (profundidade/costa)? Usa h_d2_m (batimetria do D2)
como feature única extra, com o MESMO protocolo de deixar um mês de fora e os mesmos hiperparâmetros do piloto."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from algamar_ml import config
from algamar_ml.modelos import treino as tr
cfg = config.carrega(); _, cartao = tr.carrega(cfg["caminhos"]["modelos"] / cfg["modelo"]["oficial"]); BASE = cartao["features"]
d = pd.read_parquet("resultados/piloto_croco_ml/dataset_piloto.parquet")
cr = pd.read_parquet("resultados/piloto_croco_ml/croco_mensal_piloto.parquet")[["latitude", "longitude", "h_d2_m"]].drop_duplicates(["latitude", "longitude"])
d = d.merge(cr.assign(latitude=cr.latitude.round(5), longitude=cr.longitude.round(5)), on=["latitude", "longitude"], how="left")
d = d[d.fonte_croco != "REST_JAN2024_PRE_OFICIAL"].reset_index(drop=True); p0 = float(d.floracao.mean())
mg = lambda x: tr.margem_base(x, p0)
def oof(feats):
    o = np.full(len(d), np.nan)
    for t in sorted(d.t0.unique()):
        te = (d.t0 == t).to_numpy(); m = tr.novo_modelo(cfg["modelo"]); m.set_params(early_stopping_rounds=None)
        m.fit(d.loc[~te, feats], d.floracao[~te].astype(int), base_margin=mg(d[~te])); o[te] = m.predict_proba(d.loc[te, feats], base_margin=mg(d[te]))[:, 1]
    return o
y = d.floracao.astype(int).to_numpy(); out = []
for nome, f in {"P0": BASE, "P0+profundidade": BASE + ["h_d2_m"], "P0+CROCO_TS": BASE + ["croco_temperature_surface", "croco_salinity_surface"],
                "P0+CROCO_CORE": BASE + ["croco_temperature_surface", "croco_salinity_surface", "croco_mld", "croco_n2_max"],
                "P0+profundidade+CROCO_CORE": BASE + ["h_d2_m", "croco_temperature_surface", "croco_salinity_surface", "croco_mld", "croco_n2_max"]}.items():
    p = oof(f); out.append({"modelo": nome, "pr_auc": ap(y, p), **{f"pr_auc_{t}": ap(y[d.t0 == t], p[(d.t0 == t).to_numpy()]) for t in sorted(d.mes_t0.unique()) for _ in [0] if False}})
    out[-1].update({m: ap(y[(d.mes_t0 == m).to_numpy()], p[(d.mes_t0 == m).to_numpy()]) for m in sorted(d.mes_t0.unique())})
r = pd.DataFrame(out); r.to_csv("resultados/piloto_croco_ml/controle_profundidade.csv", index=False, float_format="%.4f"); print(r.round(3).to_string())
