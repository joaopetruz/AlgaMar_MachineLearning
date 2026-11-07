"""Etapa 13.7 — validação da previsão de floração para os próximos 12 meses com o modelo oficial baseline-hist-api-v1.

Usa o modelo SALVO (baselines/baseline-hist-api-v1), sem retreino.
- Linhas: todas as emissões do teste (t0 ≥ 2022-01) com alvo observado, h = 1..12.
- Cada previsão traz prediction_date, target_date, horizon, lat, lon, probabilidade, classe, nível de risco, modelo e versão.
- Métricas por horizonte: PR-AUC, ROC-AUC, F1, recall, precision, Brier e log loss. Referências: climatologia (frequência
  histórica de floração no ponto e mês-alvo, a feature freq_floracao_alvo) e persistência (floração em t0).
- Região: os 3 setores do config, mais a habilidade por ponto.
- Localização: em cada mês-alvo, a AUC entre pontos (o modelo ordena bem ONDE floresce?), comparada com a climatologia.
- Calibração: diagrama de confiabilidade por faixa de 10 %, com destaque para ~20 %, ~50 % e ~80 %.
- Origem dos dados: t0 ≤ 2024-12 usa produtos reprocessados (MY/REP); de 2025 em diante, CHL MY e SST NRT (manifesto do painel).
Saídas: resultados/etapa13_07_validacao/
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)

from algamar_ml import config, pipeline
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import indice_mes, texto_mes
from algamar_ml.modelos import treino as tr
from algamar_ml.previsao import nivel_risco

t00 = time.time()
O = Path("resultados/etapa13_07_validacao"); O.mkdir(parents=True, exist_ok=True)
cfg = config.carrega()
VERSAO = cfg["modelo"]["oficial"]
modelo, cartao = tr.carrega(Path(cfg["caminhos"]["modelos"]) / VERSAO)
cubo, _ = pipeline.carrega_cubo(cfg)
linhas = pipeline.dataset(cfg, cubo, {})
assert set(cartao["features"]) <= set(linhas.columns)
_, _, te = tr.divide(linhas, cfg["split"])
L = linhas[te].reset_index(drop=True)
p = tr.probabilidade(modelo, L, cartao)
LA, LM = cartao["limiar_alto"], cartao["limiar_moderado"]
y = L["floracao"].to_numpy(dtype=int); h = L["h"].to_numpy()
clim = np.clip(L["freq_floracao_alvo"].fillna(cartao.get("p0", 0.19)).to_numpy(dtype=float), 1e-4, 1 - 1e-4)
pers = (L["chl_max_t0"].to_numpy() > cfg["alvo"]["limiar"]).astype(float)
reg = np.asarray(regiao(L["longitude"], cfg["regioes"])).astype(str)
fonte = np.where(L["t0"].to_numpy() <= indice_mes(2024, 12), "REPROCESSADO_2022_2024", "MY_CHL_NRT_SST_2025_2026")
print(f"linhas de teste {len(L):,} | {time.time() - t00:.0f}s", flush=True)

# ------------------------------------------------------------------ previsões (formato pedido)
prev = pd.DataFrame({"prediction_date": texto_mes(L["t0"].to_numpy()), "target_date": texto_mes(L["tau"].to_numpy()),
                     "horizon": h, "latitude": L["latitude"].round(5), "longitude": L["longitude"].round(5),
                     "probabilidade": p.round(4), "classe": (p >= LA).astype(int), "nivel_de_risco": nivel_risco(p, LA, LM),
                     "modelo": "XGBoost multi-horizonte com base climatológica", "versao": VERSAO,
                     "observado": y, "regiao": reg, "fonte_inputs": fonte})
prev.to_parquet(O / "previsoes_teste_hindcast.parquet", index=False, compression="zstd")


def met(yy, pp, lim):
    if len(np.unique(yy)) < 2:
        return {}
    c = pp >= lim
    return {"n": int(len(yy)), "taxa_obs": float(yy.mean()), "pr_auc": average_precision_score(yy, pp), "roc_auc": roc_auc_score(yy, pp),
            "f1": f1_score(yy, c), "recall": recall_score(yy, c), "precision": precision_score(yy, c, zero_division=0),
            "brier": brier_score_loss(yy, pp), "log_loss": log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6))}


# ------------------------------------------------------------------ por horizonte (+ referências e IC por blocos)
blocos = (L["tau"] // 12).astype(str).to_numpy() + "|" + reg + "|" + ((L["tau"] % 12) // 3).astype(str).to_numpy()
rng = np.random.default_rng(0)
lin = []
for k in list(range(1, 13)) + ["todos"]:
    m = np.ones(len(L), bool) if k == "todos" else h == k
    r = {"h": k, **met(y[m], p[m], LA)}
    r["pr_auc_climatologia"] = average_precision_score(y[m], clim[m]); r["brier_climatologia"] = brier_score_loss(y[m], clim[m])
    r["pr_auc_persistencia"] = average_precision_score(y[m], pers[m]); r["f1_persistencia"] = f1_score(y[m], pers[m] > 0.5)
    r["brier_skill_vs_clim"] = 1 - r["brier"] / r["brier_climatologia"]
    bb = blocos[m]; ub = np.unique(bb); ix = {b: np.where(bb == b)[0] for b in ub}; yy, pp, cc = y[m], p[m], clim[m]
    d = []
    for _ in range(300):
        a = np.concatenate([ix[b] for b in rng.choice(ub, len(ub))])
        d.append(average_precision_score(yy[a], pp[a]) - average_precision_score(yy[a], cc[a]))
    r["delta_pr_auc_vs_clim"] = r["pr_auc"] - r["pr_auc_climatologia"]
    r["ic95_lo"], r["ic95_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    lin.append(r)
por_h = pd.DataFrame(lin); por_h.to_csv(O / "metricas_por_horizonte.csv", index=False, float_format="%.5f")
print(por_h[["h", "pr_auc", "pr_auc_climatologia", "pr_auc_persistencia", "roc_auc", "f1", "recall", "precision", "brier",
             "log_loss", "brier_skill_vs_clim", "ic95_lo", "ic95_hi"]].round(3).to_string(), flush=True)

# ------------------------------------------------------------------ por fonte dos inputs (reprocessado × 2025–26)
pf = pd.DataFrame([{"fonte_inputs": f, "h": k, **met(y[(fonte == f) & (h == k)], p[(fonte == f) & (h == k)], LA),
                    "pr_auc_climatologia": average_precision_score(y[(fonte == f) & (h == k)], clim[(fonte == f) & (h == k)])}
                   for f in np.unique(fonte) for k in (1, 3, 6, 12)])
pf.to_csv(O / "metricas_por_fonte.csv", index=False, float_format="%.5f")

# ------------------------------------------------------------------ região
pr = pd.DataFrame([{"regiao": g, "h": k, **met(y[(reg == g) & (h == k)], p[(reg == g) & (h == k)], LA),
                    "pr_auc_climatologia": average_precision_score(y[(reg == g) & (h == k)], clim[(reg == g) & (h == k)])}
                   for g in np.unique(reg) for k in (1, 3, 6, 12)])
pr["delta_vs_clim"] = pr.pr_auc - pr.pr_auc_climatologia
pr.to_csv(O / "metricas_por_regiao.csv", index=False, float_format="%.5f")
print(pr[["regiao", "h", "taxa_obs", "pr_auc", "pr_auc_climatologia", "delta_vs_clim", "f1", "brier"]].round(3).to_string(), flush=True)

# habilidade por ponto (Brier skill vs climatologia) em h = 1, 3, 6, 12
pts = []
for k in (1, 3, 6, 12):
    s = prev[prev.horizon == k].assign(c=clim[h == k])
    for (la, lo), g in s.groupby(["latitude", "longitude"]):
        b, bc = brier_score_loss(g.observado, g.probabilidade), brier_score_loss(g.observado, g.c)
        pts.append({"h": k, "latitude": la, "longitude": lo, "regiao": g.regiao.iloc[0], "freq_obs": g.observado.mean(),
                    "brier": b, "brier_clim": bc, "bss": 1 - b / bc if bc > 0 else np.nan})
pts = pd.DataFrame(pts); pts.to_csv(O / "habilidade_por_ponto.csv", index=False, float_format="%.5f")

# ------------------------------------------------------------------ localização: AUC entre pontos em cada mês-alvo
loc = []
for (tau, k), g in prev.assign(c=clim, tau=L["tau"].to_numpy()).groupby(["tau", "horizon"]):
    if 0 < g.observado.sum() < len(g):
        loc.append({"tau": tau, "h": k, "auc_espacial_modelo": roc_auc_score(g.observado, g.probabilidade),
                    "auc_espacial_clim": roc_auc_score(g.observado, g.c), "frac_pontos_floracao": g.observado.mean(),
                    "precisao_alto": float(g.observado[g.probabilidade >= LA].mean()) if (g.probabilidade >= LA).any() else np.nan})
loc = pd.DataFrame(loc); loc.to_csv(O / "localizacao_por_mes.csv", index=False, float_format="%.5f")
loc_h = loc.groupby("h")[["auc_espacial_modelo", "auc_espacial_clim", "precisao_alto"]].mean()
loc_h["frac_meses_modelo_melhor"] = loc.assign(b=loc.auc_espacial_modelo > loc.auc_espacial_clim).groupby("h").b.mean()
loc_h.to_csv(O / "localizacao_por_horizonte.csv", float_format="%.5f")
print(loc_h.round(3).to_string(), flush=True)

# ------------------------------------------------------------------ calibração
bins = np.linspace(0, 1, 11)
cal = []
for k in (1, 3, 6, 12, "todos"):
    m = np.ones(len(L), bool) if k == "todos" else h == k
    b = np.clip(np.digitize(p[m], bins) - 1, 0, 9)
    for i in range(10):
        s = b == i
        if s.sum():
            cal.append({"h": k, "faixa": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "n": int(s.sum()), "prob_media": float(p[m][s].mean()),
                        "freq_observada": float(y[m][s].mean())})
    for alvo in (0.2, 0.5, 0.8):
        s = np.abs(p[m] - alvo) <= 0.05
        cal.append({"h": k, "faixa": f"~{int(alvo*100)}% (±5)", "n": int(s.sum()), "prob_media": float(p[m][s].mean()) if s.any() else np.nan,
                    "freq_observada": float(y[m][s].mean()) if s.any() else np.nan})
cal = pd.DataFrame(cal); cal.to_csv(O / "calibracao.csv", index=False, float_format="%.4f")
print(cal[cal.faixa.str.startswith("~")].round(3).to_string(), flush=True)

json.dump({"versao": VERSAO, "limiar_alto": LA, "limiar_moderado": LM, "linhas_teste": int(len(L)),
           "periodo_emissoes": [texto_mes(int(L.t0.min())), texto_mes(int(L.t0.max()))],
           "periodo_alvos": [texto_mes(int(L.tau.min())), texto_mes(int(L.tau.max()))],
           "linhas_por_fonte": pd.Series(fonte).value_counts().to_dict(), "segundos": round(time.time() - t00)},
          open(O / "resumo.json", "w"), indent=1, ensure_ascii=False)
print("ok", round(time.time() - t00), "s")
