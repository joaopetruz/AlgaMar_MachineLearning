import xarray as xr
import pandas as pd

anos = list(range(2000, 2025))
tabelas_mensais = []

precipitacao = pd.read_csv("dados/precipitacao_2000_2024_sp.csv")
# Arredondar coordenadas da precipitação para bater com a grade principal
precipitacao["latitude"] = precipitacao["latitude"].round(5)
precipitacao["longitude"] = precipitacao["longitude"].round(5)

for ano in anos:
    print(f"Processando ano {ano}...")

    clorofila = xr.open_dataset(f"dados/clorofila_{ano}_sp.nc")
    temperatura = xr.open_dataset(f"dados/temperatura_{ano}_sp.nc")
    salinidade = xr.open_dataset(f"dados/salinidade_{ano}_sp.nc")

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

    # Arredondar as coordenadas da grade principal também, para bater exatamente
    mensal["latitude"] = mensal["latitude"].round(5)
    mensal["longitude"] = mensal["longitude"].round(5)

    precip_ano = precipitacao[precipitacao["ano"] == ano]
    mensal = mensal.merge(precip_ano, on=["latitude", "longitude", "ano", "mes"], how="left")

    tabelas_mensais.append(mensal)
    print(f"Ano {ano} processado: {len(mensal)} linhas mensais")

tabela_final = pd.concat(tabelas_mensais, ignore_index=True)

print(f"\nTotal de linhas: {len(tabela_final)}")
print(f"\nValores faltando:")
print(tabela_final.isna().sum())

tabela_final = tabela_final.dropna()
print(f"\nLinhas finais: {len(tabela_final)}")

tabela_final.to_csv("dados/dados_mensais_2000_2024_sp_precipitacao.csv", index=False)
print("\nArquivo salvo: dados/dados_mensais_2000_2024_sp_precipitacao.csv")
print("\nPrimeiras linhas:")
print(tabela_final.head(10))