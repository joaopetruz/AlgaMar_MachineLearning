"""Etapa 13.2B — alternativas para dist_hotspot, comparadas SÓ na validação 2019-2021 (o teste não é olhado).

H0 coordenadas fixas originais (escolhidas visualmente num mapa de 2024, ano de teste), distância em graus
H1 as mesmas coordenadas, distância em km
H2 hotspots recalculados com regra objetiva só nos anos de treino (2000-2018): 2 maiores médias de clorofila, >= 50 km
H3 sem dist_hotspot (lat/lon continuam como features)
Todas com as features corrigidas e o protocolo novo. Saída: resultados/etapa13_02b/hotspots_validacao.json
"""
import json
from pathlib import Path

import pandas as pd

from algamar_ml.avaliacao import ANO_FIM_TREINO, divide, escolhe_limiar, metricas, novo_modelo
from algamar_ml.features import FEATURES_BASELINE, HOTSPOTS, constroi_features
from algamar_ml.features.sazonal import distancia_hotspot_km, hotspots_do_treino

mensal = pd.read_csv("dados/dados_mensais_2000_2024_sp_com_salinidade.csv")
f = constroi_features(mensal)
hs_treino = hotspots_do_treino(mensal, ano_fim=ANO_FIM_TREINO)
dist_orig_para_h2 = [float(distancia_hotspot_km(pd.DataFrame({"latitude": [a], "longitude": [b]}), hs_treino)[0]) for a, b in HOTSPOTS]
print("hotspots originais:", HOTSPOTS, "| recalculados no treino:", hs_treino, "| distância original→H2 (km):", [round(d, 1) for d in dist_orig_para_h2], flush=True)

variantes = {"H0_original_graus": f["dist_hotspot"].to_numpy(), "H1_original_km": distancia_hotspot_km(f, HOTSPOTS),
             "H2_treino_km": distancia_hotspot_km(f, hs_treino), "H3_sem_hotspot": None}
tr, va, _ = divide(f)
y = f["floracao"]
out = {}
for nome, col in variantes.items():
    X = f[FEATURES_BASELINE].copy()
    if col is None:
        X = X.drop(columns="dist_hotspot")
    else:
        X["dist_hotspot"] = col
    m = novo_modelo(y[tr])
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[va])[:, 1]
    lim = escolhe_limiar(y[va], p)
    mm = metricas(y[va], p, lim)
    out[nome] = {k: mm[k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "log_loss", "limiar")}
    print(nome, {k: round(v, 4) for k, v in out[nome].items()}, flush=True)
O = Path("resultados/etapa13_02b")
O.mkdir(parents=True, exist_ok=True)
json.dump({"conjunto": "validação 2019-2021 (treino <= 2018); teste NÃO usado", "hotspots_originais": HOTSPOTS,
           "hotspots_recalculados_treino": hs_treino, "distancia_original_para_recalculado_km": dist_orig_para_h2,
           "validacao": out}, open(O / "hotspots_validacao.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
