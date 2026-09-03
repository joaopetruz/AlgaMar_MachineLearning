import xarray as xr
import pandas as pd

anos = list(range(2000, 2025))
tabelas_mensais = []

for ano in anos:
    print(f"Processando ano {ano}...")

    clorofila = xr.open_dataset(f"dados/clorofila_{ano}_sp.nc")
    temperatura = xr.open_dataset(f"dados/temperatura_{ano}_sp.nc")

    temperatura_ajustada = temperatura.interp(
        latitude=clorofila.latitude,
        longitude=clorofila.longitude
    )

    dados = xr.merge([clorofila, temperatura_ajustada])
    dados["analysed_sst"] = dados["analysed_sst"] - 273.15

    tabela = dados.to_dataframe().reset_index()
    tabela = tabela.rename(columns={
        "CHL": "clorofila",
        "analysed_sst": "temperatura_celsius"
    })

    # Remover pontos de terra (sempre NaN)
    por_local = tabela.groupby(["latitude", "longitude"])["clorofila"].apply(lambda x: x.isna().mean())
    locais_oceano = por_local[por_local < 1.0].index
    tabela = tabela.set_index(["latitude", "longitude"])
    tabela = tabela.loc[tabela.index.isin(locais_oceano)]
    tabela = tabela.reset_index()

    # Agregar por mês (média e máximo de clorofila, média de temperatura)
    tabela["ano"] = tabela["time"].dt.year
    tabela["mes"] = tabela["time"].dt.month

    mensal = tabela.groupby(["latitude", "longitude", "ano", "mes"]).agg(
        clorofila_media=("clorofila", "mean"),
        clorofila_maxima=("clorofila", "max"),
        temperatura_media=("temperatura_celsius", "mean")
    ).reset_index()

    tabelas_mensais.append(mensal)
    print(f"Ano {ano} processado: {len(mensal)} linhas mensais")

# Juntar todos os anos
tabela_final = pd.concat(tabelas_mensais, ignore_index=True)

print(f"\nTotal de linhas (todos os anos, agregado mensal): {len(tabela_final)}")
print(f"\nValores faltando:")
print(tabela_final.isna().sum())

# Remover linhas com dados faltando (poucas, geralmente bordas)
tabela_final = tabela_final.dropna()
print(f"\nLinhas finais: {len(tabela_final)}")

tabela_final.to_csv("dados/dados_mensais_2016_2024_sp.csv", index=False)
print("\nArquivo salvo: dados/dados_mensais_2016_2024_sp.csv")

print("\nPrimeiras linhas:")
print(tabela_final.head(10))