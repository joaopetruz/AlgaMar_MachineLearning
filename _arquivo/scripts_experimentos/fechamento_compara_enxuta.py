"""Fechamento (2026-10-09) — a versão enxuta (risco-sat-clim-v1) perde algo em relação à oficial (baseline-hist-api-v1)?

Carrega as DUAS versões do registro (SHA256 conferido; nenhuma é retreinada aqui) e avalia no MESMO teste do config:
emissões ≥ 2022-01, todos os h = 1..12, as mesmas linhas. Referência: climatologia (freq_floracao_alvo).
- Métricas: PR-AUC, ROC-AUC, Brier, F1 (limiar de cada cartão, escolhido na validação).
- Diferença pareada enxuta − oficial com IC 95 %, bootstrap em blocos ano do alvo × região × trimestre (300 reamostragens),
  igual à 13.7.
- Também separa as emissões de 2025–26 (entradas NRT/interinas).
Saída: resultados/fechamento/{comparacao.csv, por_h.csv, resultados.json}
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from algamar_ml import config, pipeline, registro
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import indice_mes
from algamar_ml.modelos import treino as tr

t00 = time.time()
O = Path("resultados/fechamento"); O.mkdir(parents=True, exist_ok=True)
cfg = config.carrega()
VERSOES = {"oficial": "baseline-hist-api-v1", "enxuta": "risco-sat-clim-v1"}
cubo, _ = pipeline.carrega_cubo(cfg)
linhas = pipeline.dataset(cfg, cubo)
_, _, te = tr.divide(linhas, cfg["split"])
y = linhas["floracao"].to_numpy(dtype=int); h = linhas["h"].to_numpy()
print(f"linhas {len(linhas):,} | teste {te.sum():,} | {time.time() - t00:.0f}s", flush=True)

prob, lim, feats = {}, {}, {}
for nome, versao in VERSOES.items():
    m, cartao, _ = registro.carrega(cfg["caminhos"]["modelos"], versao)
    p = np.full(len(linhas), np.nan); p[te] = tr.probabilidade(m, linhas[te], cartao)
    prob[nome], lim[nome], feats[nome] = p, cartao["limiar_alto"], cartao["features"]
prob["climatologia"] = linhas["freq_floracao_alvo"].to_numpy(dtype=float)
lim["climatologia"] = None

reg = np.asarray(regiao(linhas["longitude"], cfg["regioes"])).astype(str)
blocos = (linhas["tau"] // 12).astype(str).to_numpy() + "|" + reg + "|" + ((linhas["tau"] % 12) // 3).astype(str).to_numpy()
rng = np.random.default_rng(0)


def amostras(mask):
    bb = blocos[mask]; ub = np.unique(bb); ix = {b: np.where(bb == b)[0] for b in ub}
    return [np.concatenate([ix[b] for b in rng.choice(ub, len(ub))]) for _ in range(300)]


def avalia(mask, rot):
    ok = mask & ~np.isnan(prob["climatologia"])
    yy = y[ok]; am = amostras(ok); ref = prob["oficial"][ok]; out = []
    for nome in prob:
        pp = prob[nome][ok]
        d = [average_precision_score(yy[a], pp[a]) - average_precision_score(yy[a], ref[a]) for a in am]
        out.append({**rot, "modelo": nome, "n": int(ok.sum()), "pr_auc": average_precision_score(yy, pp),
                    "roc_auc": roc_auc_score(yy, pp), "brier": brier_score_loss(yy, pp),
                    "f1": f1_score(yy, pp >= lim[nome]) if lim[nome] is not None else np.nan,
                    "delta_pr_auc_vs_oficial": average_precision_score(yy, pp) - average_precision_score(yy, ref),
                    "ic95_lo": float(np.percentile(d, 2.5)), "ic95_hi": float(np.percentile(d, 97.5))})
    return out


nrt = linhas["t0"].to_numpy() >= indice_mes(2025, 1)
geral = pd.DataFrame(avalia(te, {"conjunto": "teste completo"}) + avalia(te & ~nrt, {"conjunto": "emissões 2022–2024"})
                     + avalia(te & nrt, {"conjunto": "emissões 2025–26 (NRT)"}))
por_h = pd.DataFrame([r for k in range(1, 13) for r in avalia(te & (h == k), {"h": k})])
geral.to_csv(O / "comparacao.csv", index=False, float_format="%.5f"); por_h.to_csv(O / "por_h.csv", index=False, float_format="%.5f")
json.dump({"versoes": VERSOES, "linhas_teste": int(te.sum()), "limiares": lim, "features": feats,
           "retiradas": sorted(set(feats["oficial"]) - set(feats["enxuta"])), "segundos": round(time.time() - t00)},
          open(O / "resultados.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(geral.round(4).to_string())
print(por_h.pivot(index="h", columns="modelo", values="pr_auc").round(4).to_string())
print("ok", round(time.time() - t00), "s")
