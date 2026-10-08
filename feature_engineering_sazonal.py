import numpy as np
import pandas as pd

ANO_FIM_TREINO = 2021     # anos <= isto formam o treino; o resto é teste
LIMITE_FLORACAO = 1.5

tabela = pd.read_csv("dados/dados_mensais_2000_2024_sp_com_salinidade.csv")
tabela["data"] = pd.to_datetime(dict(year=tabela["ano"], month=tabela["mes"], day=1))
tabela = tabela.sort_values(["latitude", "longitude", "data"]).reset_index(drop=True)

local = ["latitude", "longitude"]

# ===== 1. Valores do mesmo mês no ano anterior (junção por data, não por posição) =====
# shift(12) assume 12 linhas = 12 meses, o que falha se algum mês estiver faltando.
anterior = tabela[local + ["data", "clorofila_media", "clorofila_maxima",
                           "temperatura_media", "salinidade_media"]].copy()
anterior["data"] = anterior["data"] + pd.DateOffset(years=1)
anterior = anterior.rename(columns={
    "clorofila_media": "clorofila_media_ano_anterior",
    "clorofila_maxima": "clorofila_maxima_ano_anterior",
    "temperatura_media": "temperatura_ano_anterior",
    "salinidade_media": "salinidade_ano_anterior",
})
tabela = tabela.merge(anterior, on=local + ["data"], how="left")

# ===== 2. Climatologia do mês, calculada SÓ com os anos de treino =====
# Antes ela usava 2000-2024, ou seja, o modelo via a média dos anos de teste.
treino = tabela["ano"] <= ANO_FIM_TREINO
climatologia = (
    tabela[treino]
    .groupby(local + ["mes"])
    .agg(
        clorofila_media_historica_mes=("clorofila_media", "mean"),
        salinidade_media_historica_mes=("salinidade_media", "mean"),
    )
    .reset_index()
)
tabela = tabela.merge(climatologia, on=local + ["mes"], how="left")

# ===== 3. Média dos ~3 anos anteriores, SEM incluir o mês atual =====
# closed="left" exclui a linha atual da janela. Antes o mês previsto entrava na média.
for coluna, nova in [("clorofila_media", "clorofila_media_3anos"),
                     ("salinidade_media", "salinidade_media_3anos")]:
    janela = (
        tabela.groupby(local)
        .rolling("1095D", on="data", closed="left", min_periods=12)[coluna]
        .mean()
        .rename(nova)
        .reset_index()          # colunas: latitude, longitude, data, <nova>
    )
    tabela = tabela.merge(janela, on=local + ["data"], how="left")

# ===== 4. Distância até o hotspot mais próximo =====
# ATENÇÃO: estes dois pontos foram identificados no mapa de 2024 (que está no
# período de teste). É um vazamento pequeno, mas ainda existe.
hotspots = [(-23.98, -46.35), (-23.45, -45.75)]
distancias = []
for lat_h, lon_h in hotspots:
    dist = np.sqrt((tabela["latitude"] - lat_h) ** 2 + (tabela["longitude"] - lon_h) ** 2)
    distancias.append(dist)
tabela["dist_hotspot"] = pd.concat(distancias, axis=1).min(axis=1)

# ===== 5. Remover linhas sem histórico =====
print(f"Linhas antes de remover NaN: {len(tabela)}")
tabela = tabela.dropna(subset=[
    "clorofila_media_ano_anterior", "temperatura_ano_anterior", "salinidade_ano_anterior",
    "clorofila_media_historica_mes", "salinidade_media_historica_mes",
])
print(f"Linhas depois: {len(tabela)}")

# ===== 6. Alvo =====
tabela["floracao"] = (tabela["clorofila_maxima"] > LIMITE_FLORACAO).astype(int)

print("\nDistribuição do target:")
print(tabela["floracao"].value_counts())
print(f"Porcentagem de floração: {tabela['floracao'].mean()*100:.2f}%")

tabela = tabela.drop(columns=["data"])
tabela.to_csv("dados/dados_features_sazonal_sp_com_salinidade.csv", index=False)
print("\nArquivo salvo: dados/dados_features_sazonal_sp_com_salinidade.csv")
