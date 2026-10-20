import pandas as pd
import numpy as np

tabela = pd.read_csv("dados/dados_tratados_2024_sp.csv", parse_dates=["time"])
tabela = tabela.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)

# ===== 1. Variáveis de calendário =====
tabela["mes"] = tabela["time"].dt.month
tabela["dia_do_ano"] = tabela["time"].dt.dayofyear

def estacao(mes):
    if mes in [12, 1, 2]:
        return "verao"
    elif mes in [3, 4, 5]:
        return "outono"
    elif mes in [6, 7, 8]:
        return "inverno"
    else:
        return "primavera"

tabela["estacao"] = tabela["mes"].apply(estacao)

# ===== 2. Lags temporais =====
grupo = tabela.groupby(["latitude", "longitude"])

tabela["clorofila_lag1"] = grupo["clorofila"].shift(1)
tabela["clorofila_lag3"] = grupo["clorofila"].shift(3)
tabela["clorofila_lag7"] = grupo["clorofila"].shift(7)
tabela["temperatura_lag1"] = grupo["temperatura_celsius"].shift(1)

# ===== 3. Médias móveis =====
tabela["clorofila_media_7d"] = grupo["clorofila"].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
tabela["clorofila_media_14d"] = grupo["clorofila"].transform(
    lambda x: x.rolling(window=14, min_periods=1).mean()
)
tabela["temperatura_media_7d"] = grupo["temperatura_celsius"].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# ===== 4. Distância até o hotspot (Santos / Baixada Santista - SP) =====
# ===== 4. Distância até o hotspot mais próximo (dois pontos identificados no mapa) =====
hotspots = [
    (-23.98, -46.35),   # Santos / Baixada Santista
    (-23.45, -45.75),   # Ilhabela / São Sebastião / Ubatuba
]

distancias = []
for lat_h, lon_h in hotspots:
    dist = np.sqrt((tabela["latitude"] - lat_h) ** 2 + (tabela["longitude"] - lon_h) ** 2)
    distancias.append(dist)

tabela["dist_hotspot"] = pd.concat(distancias, axis=1).min(axis=1)

# ===== 5. Remover linhas sem lag =====
print(f"Linhas antes de remover NaN dos lags: {len(tabela)}")
tabela = tabela.dropna(subset=["clorofila_lag1", "clorofila_lag3", "clorofila_lag7", "temperatura_lag1"])
print(f"Linhas depois: {len(tabela)}")

# ===== 6. Criar target =====
LIMITE_FLORACAO = 1.5  # ajustado para a nova região — ver observação abaixo

tabela["floracao"] = (tabela["clorofila"] > LIMITE_FLORACAO).astype(int)

print(f"\nDistribuição do target 'floracao':")
print(tabela["floracao"].value_counts())
print(f"\nPorcentagem de casos de floração: {tabela['floracao'].mean()*100:.4f}%")

print("\nColunas finais:")
print(tabela.columns.tolist())

# ===== 7. Salvar =====
tabela.to_csv("dados/dados_features_2024_sp.csv", index=False)
print("\nArquivo salvo: dados/dados_features_2024_sp.csv")