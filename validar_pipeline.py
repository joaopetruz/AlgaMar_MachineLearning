import sys

import numpy as np
import pandas as pd

MENSAL = "dados/dados_mensais_2000_2024_sp_com_salinidade.csv"
FEATURES = "dados/dados_features_sazonal_sp_com_salinidade.csv"
ANO_FIM_TREINO = 2021
SANTOS = (-23.96, -46.35)
COLUNAS_MODELO = [
    "mes", "latitude", "longitude", "dist_hotspot",
    "clorofila_media_ano_anterior", "clorofila_maxima_ano_anterior",
    "temperatura_ano_anterior", "salinidade_ano_anterior",
    "clorofila_media_historica_mes", "salinidade_media_historica_mes",
    "clorofila_media_3anos", "salinidade_media_3anos",
]

resultados = []


def checar(nome, ok, detalhe=""):
    resultados.append(ok)
    print(f"[{'OK' if ok else 'FALHA'}] {nome}" + (f" - {detalhe}" if detalhe else ""))


def aviso(nome, detalhe):
    print(f"[AVISO] {nome} - {detalhe}")


mensal = pd.read_csv(MENSAL)
feat = pd.read_csv(FEATURES)
mensal["data"] = pd.to_datetime(dict(year=mensal["ano"], month=mensal["mes"], day=1))
local = ["latitude", "longitude"]

print(f"Mensal: {len(mensal)} linhas | Features: {len(feat)} linhas\n")

# ===== 1. Estrutura =====
faltando = [c for c in COLUNAS_MODELO + ["floracao", "ano"] if c not in feat.columns]
checar("Colunas do modelo presentes", not faltando, f"faltam: {faltando}" if faltando else "")

nan_features = feat[COLUNAS_MODELO].isna().sum()
nan_features = nan_features[nan_features > 0]
if len(nan_features):
    aviso("NaN nas features (o XGBoost aceita, mas confira)", nan_features.to_dict())
else:
    checar("Sem NaN nas features do modelo", True)

# ===== 2. Vazamento: recalcular à mão numa amostra =====
rng = np.random.default_rng(0)
amostra = feat.sample(min(200, len(feat)), random_state=0)
grupos = {k: g.set_index("data") for k, g in mensal.groupby(local)}
feat["data"] = pd.to_datetime(dict(year=feat["ano"], month=feat["mes"], day=1))
amostra = feat.loc[amostra.index]

erros_3anos = erros_clim = erros_ant = 0
for _, r in amostra.iterrows():
    g = grupos[(r["latitude"], r["longitude"])]

    janela = g[(g.index < r["data"]) & (g.index >= r["data"] - pd.Timedelta(days=1095))]
    if len(janela) >= 12:
        if not np.isclose(r["clorofila_media_3anos"], janela["clorofila_media"].mean(), equal_nan=False):
            erros_3anos += 1

    treino = g[(g["ano"] <= ANO_FIM_TREINO) & (g["mes"] == r["mes"])]
    if len(treino) and not np.isclose(r["clorofila_media_historica_mes"], treino["clorofila_media"].mean()):
        erros_clim += 1

    ant = g[g.index == r["data"] - pd.DateOffset(years=1)]
    if len(ant) and not np.isclose(r["clorofila_media_ano_anterior"], ant["clorofila_media"].iloc[0]):
        erros_ant += 1

checar("Média de 3 anos não inclui o mês atual", erros_3anos == 0, f"{erros_3anos} divergências em {len(amostra)} linhas")
checar("Climatologia usa só anos de treino", erros_clim == 0, f"{erros_clim} divergências")
checar("Ano anterior casa por data", erros_ant == 0, f"{erros_ant} divergências")

# ===== 3. Cobertura espacial =====
pts_mensal = mensal[local].drop_duplicates()
pts_feat = feat[local].drop_duplicates()
perdidos = len(pts_mensal) - len(pts_feat)
print(f"\nPontos únicos: mensal={len(pts_mensal)} -> features={len(pts_feat)} (perdidos: {perdidos})")
if perdidos > 0:
    aviso("Pontos perdidos entre mensal e features",
          "provavelmente só têm dados em anos recentes (sem histórico para ano anterior/climatologia)")

km = np.hypot((pts_feat["longitude"] - SANTOS[1]) * 111 * np.cos(np.radians(SANTOS[0])),
              (pts_feat["latitude"] - SANTOS[0]) * 111)
checar("Santos coberto (ponto a menos de 5 km)", km.min() < 5, f"mais próximo a {km.min():.1f} km")
print(f"Latitude {feat.latitude.min():.3f} a {feat.latitude.max():.3f} | "
      f"Longitude {feat.longitude.min():.3f} a {feat.longitude.max():.3f}")

# ===== 4. Meses faltando por ponto =====
esperado = mensal["data"].nunique()
por_ponto = mensal.groupby(local)["data"].nunique()
falta = 1 - por_ponto / esperado
checar("Séries mensais quase completas (mediana de meses faltando < 5%)",
       falta.median() < 0.05, f"mediana {falta.median()*100:.1f}%, pior ponto {falta.max()*100:.1f}%")

# ===== 5. Salinidade preenchida =====
if "salinidade_preenchida" in mensal.columns:
    por_ano = mensal.groupby("ano")["salinidade_preenchida"].mean() * 100
    total = mensal["salinidade_preenchida"].mean() * 100
    print(f"\nSalinidade preenchida: {total:.1f}% das linhas (por ano: min {por_ano.min():.1f}%, max {por_ano.max():.1f}%)")
    if por_ano.max() > 2 * max(por_ano.median(), 1):
        aviso("Preenchimento muito desigual entre anos", "confira a cobertura do satélite de salinidade nesses anos")
else:
    aviso("Coluna salinidade_preenchida ausente", "o processar_dados_historicos.py novo não foi usado?")

# ===== 6. Estabilidade do alvo =====
taxa = feat.groupby("ano")["floracao"].mean() * 100
print("\nTaxa de floração por ano (%):")
print(taxa.round(1).to_string())
treino_mask = feat["ano"] <= ANO_FIM_TREINO
t_tr, t_te = feat.loc[treino_mask, "floracao"].mean() * 100, feat.loc[~treino_mask, "floracao"].mean() * 100
print(f"Treino: {t_tr:.1f}% | Teste: {t_te:.1f}% | linhas treino/teste: {treino_mask.sum()}/{(~treino_mask).sum()}")
checar("Taxa de floração do teste parecida com a do treino (diferença < 10 pontos)",
       abs(t_tr - t_te) < 10, f"diferença {abs(t_tr - t_te):.1f} pontos")

# ===== Resumo =====
print(f"\n{sum(resultados)} de {len(resultados)} checagens OK")
sys.exit(0 if all(resultados) else 1)
