"""Etapa 13.3 — prova empírica do horizonte do E0_original e do E0_corrigido (o que cada linha T realmente usa).

Método: para uma amostra de linhas (ponto, T), muda os dados de UM mês M = T-k (k = 0, 1, 2, ..., 13) só naquele
ponto e verifica se alguma feature da linha T muda. O maior k para o qual nada muda... ao contrário: o MENOR k cujo
dado afeta as features é o "último mês usado". Se k = 0 afeta, o modelo usa o próprio mês-alvo (NOWCAST);
se o menor k é 1, ele precisa do mês anterior (FORECAST de no máximo 1 mês); etc.
Saída: resultados/etapa13_03/horizonte_e0.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml.features import CHAVE, FEATURES_BASELINE, constroi_features, constroi_features_original

mensal = pd.read_csv("dados/dados_mensais_2000_2024_sp_com_salinidade.csv")
rng = np.random.default_rng(0)
pontos = mensal[["latitude", "longitude"]].drop_duplicates().sample(3, random_state=0).to_numpy()
VARS = ["clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"]
# subconjunto: os 3 pontos + 40 vizinhos (as features são por ponto; hotspots H2 dependem do conjunto todo)
sub = mensal[mensal.set_index(["latitude", "longitude"]).index.isin([tuple(p) for p in pontos])].copy()
alvos = [(2020, 7), (2023, 1)]
res = {}
for nome, construtor in (("E0_original", constroi_features_original), ("E0_corrigido", constroi_features)):
    base = construtor(sub).set_index(CHAVE)
    ks = {}
    for k in range(0, 14):
        afetou = []
        for lat, lon in pontos:
            for ano, mes in alvos:
                t = ano * 12 + mes - 1 - k
                ma, mm = divmod(t, 12)
                alt = sub.copy()
                sel = (alt.latitude == lat) & (alt.longitude == lon) & (alt.ano == ma) & (alt.mes == mm + 1)
                alt.loc[sel, VARS] = alt.loc[sel, VARS] * 3.0 + 5.0
                f = construtor(alt).set_index(CHAVE)
                chave = (lat, lon, ano, mes)
                if chave not in base.index or chave not in f.index:
                    continue
                dif = [c for c in FEATURES_BASELINE if c not in CHAVE and not np.isclose(base.loc[chave, c], f.loc[chave, c], equal_nan=True)]
                afetou += dif
        ks[k] = sorted(set(afetou))
    menor = min([k for k, v in ks.items() if v], default=None)
    res[nome] = {"features_afetadas_por_mudar_o_mes_T_menos_k": {str(k): v for k, v in ks.items()},
                 "ultimo_mes_usado": f"T-{menor}" if menor is not None else "nenhum",
                 "classe": {0: "NOWCAST (usa o próprio mês-alvo)", 1: "FORECAST_1M no máximo (precisa do mês T-1)"}.get(menor, f"FORECAST até {menor} meses")}
    print(nome, res[nome]["ultimo_mes_usado"], res[nome]["classe"])
    for k in range(0, 14):
        if ks[k]:
            print(f"  muda o mês T-{k}: {ks[k]}")
O = Path("resultados/etapa13_03")
O.mkdir(parents=True, exist_ok=True)
json.dump({"metodo": __doc__, "linhas_testadas": [list(map(float, p)) + [a, m] for p in pontos for a, m in alvos], **res},
          open(O / "horizonte_e0.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
