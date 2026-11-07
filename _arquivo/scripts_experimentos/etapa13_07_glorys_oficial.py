"""Etapa 13.7 — o GLORYS melhora o modelo de previsão OFICIAL (baseline-hist-api-v1)?

Treina o modelo oficial com o MESMO config (hiperparâmetros, base climatológica, split, seed) em três variantes:
  OF0  oficial (features do hist-api-v1)
  OF1  OF0 + MLD GLORYS em t0
  OF2  OF0 + T/S + MLD GLORYS em t0
Todas usam as mesmas linhas. A avaliação usa o teste do config (emissões ≥ 2022-01), restrito a t0 ≤ 2024-12, o último mês
do GLORYS mensal baixado, para que as três variantes sejam comparadas onde a física existe.
- Métricas: PR-AUC, ROC-AUC, Brier e F1 (limiar = máximo F1 na validação), geral e por horizonte.
- Diferença pareada contra OF0 com IC 95 %, por bootstrap em blocos ano do alvo × região × trimestre (300 reamostragens).
Saída: resultados/etapa13_07/{comparacao.csv, por_h.csv, resultados.json}
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from algamar_ml import config, glorys_ml, metricas, pipeline
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import indice_mes
from algamar_ml.modelos import treino as tr

t00 = time.time()
O = Path("resultados/etapa13_07"); O.mkdir(parents=True, exist_ok=True)
cfg = config.carrega()
cubo, _ = pipeline.carrega_cubo(cfg)
fis = glorys_ml.para_cubo(cfg["caminhos"]["glorys_mensal"], cubo, {"use_glorys_ts": True, "use_glorys_mld": True})
linhas = pipeline.dataset(cfg, cubo, fis)
base = pipeline.features_de(cfg, [])
VAR = {"OF0_oficial": base, "OF1_+MLD_GLORYS": base + ["glorys_mld_t0"],
       "OF2_+TS_MLD_GLORYS": base + ["glorys_ts_t0", "glorys_ss_t0", "glorys_mld_t0"]}
tr_, va, te = tr.divide(linhas, cfg["split"])
te = te & (linhas["t0"].to_numpy() <= indice_mes(2024, 12))
y = linhas["floracao"].to_numpy(dtype=int); h = linhas["h"].to_numpy()
print(f"linhas {len(linhas):,} | teste (t0 ≤ 2024-12) {te.sum():,} | {time.time() - t00:.0f}s", flush=True)
assert linhas.loc[te, "glorys_mld_t0"].notna().mean() > 0.99

prob, lim = {}, {}
for nome, feats in VAR.items():
    m = tr.treina(linhas, feats, cfg)
    uso = {"features": feats, "base_climatologica": bool(cfg["modelo"].get("base_climatologica", False)), "p0": m.algamar_p0}
    lim[nome] = metricas.escolhe_limiar(y[va], tr.probabilidade(m, linhas[va], uso))
    p = np.full(len(linhas), np.nan); p[te] = tr.probabilidade(m, linhas[te], uso); prob[nome] = p
    print(nome, f"{time.time() - t00:.0f}s", flush=True)

reg = np.asarray(regiao(linhas["longitude"], cfg["regioes"])).astype(str)
blocos = (linhas["tau"] // 12).astype(str).to_numpy() + "|" + reg + "|" + ((linhas["tau"] % 12) // 3).astype(str).to_numpy()
rng = np.random.default_rng(0)


def amostras(mask):
    bb = blocos[mask]; ub = np.unique(bb); ix = {b: np.where(bb == b)[0] for b in ub}
    return [np.concatenate([ix[b] for b in rng.choice(ub, len(ub))]) for _ in range(300)]


def avalia(mask, rot):
    yy = y[mask]; am = amostras(mask); ref = prob["OF0_oficial"][mask]; out = []
    for nome in VAR:
        pp = prob[nome][mask]
        d = [average_precision_score(yy[a], pp[a]) - average_precision_score(yy[a], ref[a]) for a in am]
        out.append({**rot, "modelo": nome, "n": int(mask.sum()), "pr_auc": average_precision_score(yy, pp), "roc_auc": roc_auc_score(yy, pp),
                    "brier": brier_score_loss(yy, pp), "f1": f1_score(yy, pp >= lim[nome]),
                    "delta_pr_auc_vs_OF0": average_precision_score(yy, pp) - average_precision_score(yy, ref),
                    "ic95_lo": float(np.percentile(d, 2.5)), "ic95_hi": float(np.percentile(d, 97.5)), "frac_boot_pos": float(np.mean(np.array(d) > 0))})
    return out


geral = pd.DataFrame(avalia(te, {"h": "todos"}))
por_h = pd.DataFrame([r for k in range(1, 13) for r in avalia(te & (h == k), {"h": k})])
geral.to_csv(O / "comparacao.csv", index=False, float_format="%.5f"); por_h.to_csv(O / "por_h.csv", index=False, float_format="%.5f")
json.dump({"linhas": int(len(linhas)), "teste": "emissões 2022-01 a 2024-12 (GLORYS mensal até 2024-12)", "limiares": lim,
           "features": VAR, "segundos": round(time.time() - t00)}, open(O / "resultados.json", "w"), indent=1, ensure_ascii=False)
print(geral.round(4).to_string())
print(por_h.pivot(index="h", columns="modelo", values="delta_pr_auc_vs_OF0").round(3).to_string())
print("ok", round(time.time() - t00), "s")
