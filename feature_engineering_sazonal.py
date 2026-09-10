import pandas as pd
import numpy as np

tabela = pd.read_csv("dados/dados_mensais_2000_2024_sp_completo.csv")
tabela = tabela.sort_values(["latitude", "longitude", "ano", "mes"]).reset_index(drop=True)

grupo = tabela.groupby(["latitude", "longitude"])

# ===== Lags anuais =====
tabela["clorofila_media_ano_anterior"] = grupo["clorofila_media"].shift(12)
tabela["clorofila_maxima_ano_anterior"] = grupo["clorofila_maxima"].shift(12)
tabela["temperatura_ano_anterior"] = grupo["temperatura_media"].shift(12)
tabela["salinidade_ano_anterior"] = grupo["salinidade_media"].shift(12)
tabela["vento_velocidade_ano_anterior"] = grupo["vento_velocidade"].shift(12)

# ===== Climatologia (média histórica daquele mês) =====
tabela["clorofila_media_historica_mes"] = tabela.groupby(["latitude", "longitude", "mes"])["clorofila_media"].transform("mean")
tabela["salinidade_media_historica_mes"] = tabela.groupby(["latitude", "longitude", "mes"])["salinidade_media"].transform("mean")
tabela["vento_media_historica_mes"] = tabela.groupby(["latitude", "longitude", "mes"])["vento_velocidade"].transform("mean")

# ===== Tendência (últimos 3 anos) =====
tabela["clorofila_media_3anos"] = grupo["clorofila_media"].transform(lambda x: x.rolling(window=36, min_periods=12).mean())
tabela["salinidade_media_3anos"] = grupo["salinidade_media"].transform(lambda x: x.rolling(window=36, min_periods=12).mean())
tabela["vento_media_3anos"] = grupo["vento_velocidade"].transform(lambda x: x.rolling(window=36, min_periods=12).mean())

# ===== Distância dos hotspots =====
hotspots = [(-23.98, -46.35), (-23.45, -45.75)]
distancias = []
for lat_h, lon_h in hotspots:
    dist = np.sqrt((tabela["latitude"] - lat_h) ** 2 + (tabela["longitude"] - lon_h) ** 2)
    distancias.append(dist)
tabela["dist_hotspot"] = pd.concat(distancias, axis=1).min(axis=1)

# ===== Remover linhas sem lag =====
print(f"Linhas antes de remover NaN: {len(tabela)}")
tabela = tabela.dropna(subset=[
    "clorofila_media_ano_anterior", "temperatura_ano_anterior",
    "salinidade_ano_anterior", "vento_velocidade_ano_anterior"
])
print(f"Linhas depois: {len(tabela)}")

# ===== Target =====
LIMITE_FLORACAO = 1.5
tabela["floracao"] = (tabela["clorofila_maxima"] > LIMITE_FLORACAO).astype(int)

print(f"\nDistribuição do target:")
print(tabela["floracao"].value_counts())
print(f"Porcentagem de floração: {tabela['floracao'].mean()*100:.2f}%")

tabela.to_csv("dados/dados_features_sazonal_sp_completo.csv", index=False)
print("\nArquivo salvo: dados/dados_features_sazonal_sp_completo.csv")
print("\nPrimeiras linhas:")
print(tabela.head(5))