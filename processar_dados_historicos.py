import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

anos = list(range(2000, 2025))
tabelas_mensais = []

for ano in anos:
    print(f"Processando ano {ano}...")

    clorofila = xr.open_dataset(f"dados/clorofila_{ano}_sp.nc")
    temperatura = xr.open_dataset(f"dados/temperatura_{ano}_sp.nc")
    salinidade = xr.open_dataset(f"dados/salinidade_{ano}_sp.nc")

    if ano == anos[0]:
        print("Primeiro timestamp -> clorofila:", clorofila.time.values[0],
              "| temperatura:", temperatura.time.values[0],
              "| salinidade:", salinidade.time.values[0])

    temperatura_ajustada = temperatura.interp(latitude=clorofila.latitude, longitude=clorofila.longitude)
    salinidade_ajustada = salinidade.interp(latitude=clorofila.latitude, longitude=clorofila.longitude)

    dados = xr.merge([clorofila, temperatura_ajustada, salinidade_ajustada])
    dados["analysed_sst"] = dados["analysed_sst"] - 273.15

    tabela = dados.to_dataframe().reset_index()
    tabela = tabela.rename(columns={
        "CHL": "clorofila",
        "analysed_sst": "temperatura_celsius",
        "sos": "salinidade"
    })

    por_local = tabela.groupby(["latitude", "longitude"])["clorofila"].apply(lambda x: x.isna().mean())
    locais_oceano = por_local[por_local < 1.0].index
    tabela = tabela.set_index(["latitude", "longitude"])
    tabela = tabela.loc[tabela.index.isin(locais_oceano)]
    tabela = tabela.reset_index()

    tabela["ano"] = tabela["time"].dt.year
    tabela["mes"] = tabela["time"].dt.month

    mensal = tabela.groupby(["latitude", "longitude", "ano", "mes"]).agg(
        clorofila_media=("clorofila", "mean"),
        clorofila_maxima=("clorofila", "max"),
        temperatura_media=("temperatura_celsius", "mean"),
        salinidade_media=("salinidade", "mean")
    ).reset_index()

    tabelas_mensais.append(mensal)
    print(f"Ano {ano} processado: {len(mensal)} linhas mensais")

tabela_final = pd.concat(tabelas_mensais, ignore_index=True)

# O satélite de salinidade mascara a faixa costeira. Sem tratamento, o dropna()
# abaixo apaga esses pontos (ex.: Santos fica a 11 km do dado mais próximo).
# Aqui a salinidade que falta recebe o valor do ponto válido mais próximo no
# mesmo mês, desde que esteja a até LIMITE_PREENCHIMENTO_GRAUS de distância.
LIMITE_PREENCHIMENTO_GRAUS = 0.5
tabela_final["salinidade_preenchida"] = False

for _, indices in tabela_final.groupby(["ano", "mes"]).groups.items():
    grupo = tabela_final.loc[indices]
    validos = grupo[grupo["salinidade_media"].notna()]
    faltando = grupo[grupo["salinidade_media"].isna()]
    if validos.empty or faltando.empty:
        continue

    arvore = cKDTree(validos[["latitude", "longitude"]].to_numpy())
    dist, pos = arvore.query(
        faltando[["latitude", "longitude"]].to_numpy(),
        distance_upper_bound=LIMITE_PREENCHIMENTO_GRAUS,
    )
    achou = np.isfinite(dist)
    destino = faltando.index[achou]
    tabela_final.loc[destino, "salinidade_media"] = validos["salinidade_media"].to_numpy()[pos[achou]]
    tabela_final.loc[destino, "salinidade_preenchida"] = True

print(f"\nLinhas com salinidade preenchida: {int(tabela_final['salinidade_preenchida'].sum())}")

print(f"\nTotal de linhas: {len(tabela_final)}")
print("\nValores faltando:")
print(tabela_final.isna().sum())

pontos_antes = tabela_final[["latitude", "longitude"]].drop_duplicates().shape[0]
tabela_final = tabela_final.dropna()
pontos_depois = tabela_final[["latitude", "longitude"]].drop_duplicates().shape[0]
print(f"\nLinhas finais: {len(tabela_final)}")
print(f"Pontos únicos: {pontos_antes} -> {pontos_depois} (perdidos no dropna: {pontos_antes - pontos_depois})")

# Nome que o feature_engineering_sazonal.py espera ler
tabela_final.to_csv("dados/dados_mensais_2000_2024_sp_com_salinidade.csv", index=False)
print("\nArquivo salvo: dados/dados_mensais_2000_2024_sp_com_salinidade.csv")
print("\nPrimeiras linhas:")
print(tabela_final.head(10))
