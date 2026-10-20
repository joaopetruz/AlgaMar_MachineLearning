"""Etapa 13.3 — E0_forecast: o baseline corrigido (13.2B) transformado em previsão FUTURA de 1 a 12 meses.

Cada linha = (ponto, mês de emissão t0, horizonte h); features só com meses <= t0 (E0_FORECAST_FEATURES);
alvo = floração (clorofila_maxima > 1,5) no mês t0 + h. Mesmo algoritmo e hiperparâmetros do E0 (XGBoost 300/6/0,1,
scale_pos_weight do treino, seed 42), mesmos dados (CSV mensal da equipe 2000-2024, f58dcfa8…), hotspots H2.

Split (config/ml.toml [split], por emissão e alvo; nenhum alvo em dois conjuntos):
  treino: τ <= 2018-12 | validação: t0 >= 2019-01 e τ <= 2021-12 | teste: t0 >= 2022-01 (τ <= 2024-12)
Arquitetura escolhida SÓ na validação (regra fixada antes: maior PR-AUC médio nos 12 horizontes):
  A) um modelo direto para todos os horizontes, com h como feature;  B) 12 modelos diretos, um por horizonte.
Limiar por horizonte: máximo F1 na validação, congelado antes do teste.
Demonstração real: emissão em dez/2024 → jan…dez/2025, conferida depois com a clorofila observada de 2025
(painel MY, dados/processados/painel_mensal.parquet), que o modelo nunca viu.
Saídas: resultados/etapa13_03/, modelos/e0-forecast-v1/, previsoes/e0-forecast-v1_2024-12/
"""
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from algamar_ml import config
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import E0_FORECAST_FEATURES, constroi_linhas, cubo_do_painel, indice_mes, texto_mes
from algamar_ml.features.sazonal import ANO_FIM_HOTSPOT, hotspots_do_treino
from algamar_ml.metricas import escolhe_limiar
from algamar_ml.modelos.treino import divide

t00 = time.time()
cfg = config.carrega()
O = Path("resultados/etapa13_03"); O.mkdir(parents=True, exist_ok=True)
M = Path("modelos/e0-forecast-v1"); M.mkdir(parents=True, exist_ok=True)
PV = Path("previsoes/e0-forecast-v1_2024-12"); PV.mkdir(parents=True, exist_ok=True)
ENTRADA = Path(cfg["caminhos"]["historico_equipe"])
LIMIAR_ALVO = cfg["alvo"]["limiar"]
FEAT = E0_FORECAST_FEATURES
PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss", tree_method="hist")
MANTER = ["point_id", "t0", "tau", "floracao", "freq_floracao_alvo"] + FEAT


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def metr(y, p, lim):
    y = np.asarray(y, int); p = np.asarray(p, float)
    pred = (p >= lim).astype(int)
    return {"n": int(len(y)), "positivos": int(y.sum()), "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
            "limiar": lim}


mensal = pd.read_csv(ENTRADA)
cubo = cubo_do_painel(mensal)
hs = hotspots_do_treino(mensal, ano_fim=ANO_FIM_HOTSPOT)
emissoes = range(int(indice_mes(2001, 1)), cubo.t_fim + 1)
linhas = constroi_linhas(cubo, LIMIAR_ALVO, emissoes, range(1, 13), hotspots=hs)
linhas = linhas[MANTER]
obs = linhas[linhas["floracao"].notna()].reset_index(drop=True)
tr, va, te = divide(obs, cfg["split"])
print(f"linhas observadas {len(obs):,} | treino {tr.sum():,} validação {va.sum():,} teste {te.sum():,} | {time.time()-t00:.0f}s", flush=True)
y = obs["floracao"].astype(int)
h = obs["h"].to_numpy()

# ---------------------------------------------------------------- arquiteturas (decisão só na validação)
def peso(mask):
    return float((y[mask] == 0).sum() / (y[mask] == 1).sum())

modelo_A = XGBClassifier(scale_pos_weight=peso(tr), **PARAMS).fit(obs.loc[tr, FEAT], y[tr])
pA = np.full(len(obs), np.nan); pA[va | te] = modelo_A.predict_proba(obs.loc[va | te, FEAT])[:, 1]
print(f"A treinado {time.time()-t00:.0f}s", flush=True)
modelos_B, pB = {}, np.full(len(obs), np.nan)
for k in range(1, 13):
    mk = tr & (h == k)
    modelos_B[k] = XGBClassifier(scale_pos_weight=peso(mk), **PARAMS).fit(obs.loc[mk, FEAT], y[mk])
    sel = (va | te) & (h == k)
    pB[sel] = modelos_B[k].predict_proba(obs.loc[sel, FEAT])[:, 1]
print(f"B treinado {time.time()-t00:.0f}s", flush=True)
val_arq = {}
for nome, p in (("A_um_modelo_h_feature", pA), ("B_12_modelos", pB)):
    val_arq[nome] = {str(k): float(average_precision_score(y[va & (h == k)], p[va & (h == k)])) for k in range(1, 13)}
    val_arq[nome]["media"] = float(np.mean([val_arq[nome][str(k)] for k in range(1, 13)]))
escolhida = max(val_arq, key=lambda n: val_arq[n]["media"])
p = pA if escolhida.startswith("A") else pB
print("validação PR-AUC média:", {n: round(v["media"], 4) for n, v in val_arq.items()}, "→", escolhida, flush=True)

# ---------------------------------------------------------------- limiares por horizonte (validação) e teste
limiares, por_h = {}, {}
for k in range(1, 13):
    sv, st = va & (h == k), te & (h == k)
    limiares[k] = escolhe_limiar(y[sv].to_numpy(), p[sv])
    clim = obs["freq_floracao_alvo"].to_numpy(float)
    ok = ~np.isnan(clim[st])
    por_h[k] = {"validacao": metr(y[sv], p[sv], limiares[k]), "teste": metr(y[st], p[st], limiares[k]),
                "teste_climatologia_pr_auc": float(average_precision_score(y[st][ok], clim[st][ok])),
                "periodo_alvos_teste": [str(texto_mes(obs.loc[st, "tau"].min())), str(texto_mes(obs.loc[st, "tau"].max()))]}
tab = pd.DataFrame([{"h": k, **{f"{m}": por_h[k]["teste"][m] for m in ("n", "positivos", "pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "log_loss", "limiar")},
                     "pr_auc_climatologia": por_h[k]["teste_climatologia_pr_auc"], "pr_auc_validacao": por_h[k]["validacao"]["pr_auc"]}
                    for k in range(1, 13)])
tab.to_csv(O / "metricas_por_horizonte_teste.csv", index=False, float_format="%.4f")
print(tab.round(3).to_string(index=False), flush=True)

# ---------------------------------------------------------------- salvar modelo e cartão
if escolhida.startswith("A"):
    modelo_A.save_model(M / "modelo.json")
else:
    for k, mk in modelos_B.items():
        mk.save_model(M / f"modelo_h{k:02d}.json")
cartao = {"versao": "e0-forecast-v1", "tipo": f"previsão direta multi-horizonte, arquitetura {escolhida}",
          "alvo": cfg["alvo"], "features": FEAT, "hotspots_H2": hs, "hiperparametros": PARAMS,
          "scale_pos_weight": "negativos/positivos do treino (por horizonte em B)", "split": cfg["split"],
          "limiares_por_horizonte": limiares, "regra_limiar": "máximo F1 na validação, por horizonte",
          "risco": {"ALTO": "prob >= limiar do horizonte", "MODERADO": f"prob >= {cfg['previsao']['fracao_moderado']} × limiar", "BAIXO": "resto"},
          "validacao_arquiteturas_pr_auc": val_arq}
(M / "cartao.json").write_text(json.dumps(cartao, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

# ---------------------------------------------------------------- previsão real: emissão em dez/2024
t0 = cubo.t_fim
fut = constroi_linhas(cubo, LIMIAR_ALVO, [t0], range(1, 13), exige_estado=False, hotspots=hs)
if escolhida.startswith("A"):
    pf = modelo_A.predict_proba(fut[FEAT])[:, 1]
else:
    pf = np.empty(len(fut))
    for k in range(1, 13):
        s = fut["h"].to_numpy() == k
        pf[s] = modelos_B[k].predict_proba(fut.loc[s, FEAT])[:, 1]
lim_f = fut["h"].map(limiares).to_numpy()
prev = pd.DataFrame({
    "prediction_date": pd.Timestamp.now(tz="UTC").date().isoformat(), "issue_month": texto_mes(fut["t0"].to_numpy()),
    "target_date": texto_mes(fut["tau"].to_numpy()), "forecast_horizon_months": fut["h"].astype(int),
    "point_id": fut["point_id"].astype(int), "latitude": fut["latitude"].astype(float).round(5),
    "longitude": fut["longitude"].astype(float).round(5), "region": regiao(fut["longitude"], cfg["regioes"]),
    "bloom_probability": pf.round(4), "predicted_class": (pf >= lim_f).astype(int),
    "risk_level": np.where(pf >= lim_f, "ALTO", np.where(pf >= cfg["previsao"]["fracao_moderado"] * lim_f, "MODERADO", "BAIXO")),
    "decision_threshold": lim_f, "climatological_probability": fut["freq_floracao_alvo"].astype(float).round(4),
    "model_version": "e0-forecast-v1"})
prev.to_parquet(PV / "previsao_pontos.parquet", index=False)
prev.to_csv(PV / "previsao_pontos.csv", index=False)
reg = prev.groupby(["region", "target_date", "forecast_horizon_months"]).agg(
    prob_media=("bloom_probability", "mean"), frac_alto=("risk_level", lambda s: float((s == "ALTO").mean())), pontos=("point_id", "count")).reset_index()
reg.to_csv(PV / "resumo_regioes.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------- conferência com 2025 observado (nunca visto)
painel = pd.read_parquet(cfg["caminhos"]["painel"])
p25 = painel[painel.ano == 2025][["latitude", "longitude", "ano", "mes", "clorofila_maxima"]].copy()
p25["target_date"] = p25.ano.astype(str) + "-" + p25.mes.astype(str).str.zfill(2)
p25["floracao_obs"] = (p25.clorofila_maxima > LIMIAR_ALVO).astype(float)
p25.loc[p25.clorofila_maxima.isna(), "floracao_obs"] = np.nan
for c in ("latitude", "longitude"):
    p25[c] = p25[c].round(5)
conf = prev.merge(p25[["latitude", "longitude", "target_date", "floracao_obs"]], on=["latitude", "longitude", "target_date"], how="left")
conf_h = []
for k in range(1, 13):
    s = (conf.forecast_horizon_months == k) & conf.floracao_obs.notna()
    yy, pp = conf.loc[s, "floracao_obs"].astype(int), conf.loc[s, "bloom_probability"]
    cl = conf.loc[s, "climatological_probability"]
    conf_h.append({"h": k, "target_date": conf.loc[s, "target_date"].iloc[0] if s.any() else None,
                   **(metr(yy, pp, limiares[k]) if s.any() and yy.nunique() > 1 else {"n": int(s.sum())}),
                   "pr_auc_climatologia": float(average_precision_score(yy, cl.fillna(cl.mean()))) if s.any() and yy.nunique() > 1 else None})
pd.DataFrame(conf_h).to_csv(O / "previsao_2025_x_observado.csv", index=False, float_format="%.4f")
print(pd.DataFrame(conf_h)[["h", "target_date", "n", "pr_auc", "pr_auc_climatologia", "f1"]].round(3).to_string(index=False), flush=True)

commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
manifest = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "seed": 42, "commit_base": commit,
            "dataset": {"arquivo": str(ENTRADA), "sha256": sha(ENTRADA)},
            "painel_conferencia_2025": {"arquivo": cfg["caminhos"]["painel"], "sha256": sha(cfg["caminhos"]["painel"])},
            "linhas": {"treino": int(tr.sum()), "validacao": int(va.sum()), "teste": int(te.sum())},
            "arquitetura_escolhida": escolhida, "tempo_s": round(time.time() - t00, 1)}
json.dump({"manifest": manifest, "cartao": cartao, "por_horizonte": por_h}, open(O / "resultados_e0_forecast.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
print("ok", manifest)
