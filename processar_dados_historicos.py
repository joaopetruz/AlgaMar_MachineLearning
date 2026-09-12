import xarray as xr
import pandas as pd
import numpy as np

anos = list(range(2000, 2025))
tabelas_mensais = []

for ano in anos:
    print(f"Processando ano {ano}...")

    clorofila = xr.open_dataset(f"dados/clorofila_{ano}_sp.nc")
    temperatura = xr.open_dataset(f"dados/temperatura_{ano}_sp.nc")
    salinidade = xr.open_dataset(f"dados/salinidade_{ano}_sp.nc")
    vento = xr.open_dataset(f"dados/vento_{ano}_sp.nc")
    correntes = xr.open_dataset(f"dados/correntes_{ano}_sp.nc")

    # Remover dimensão de profundidade das correntes (pegamos só superfície)
    if "depth" in correntes.dims:
        correntes = correntes.isel(depth=0)

    temperatura_ajustada = temperatura.interp(latitude=clorofila.latitude, longitude=clorofila.longitude)
    salinidade_ajustada = salinidade.interp(latitude=clorofila.latitude, longitude=clorofila.longitude)
    vento_ajustado = vento.interp(latitude=clorofila.latitude, longitude=clorofila.longitude)
    correntes_ajustadas = correntes.interp(latitude=clorofila.latitude, longitude=clorofila.longitude)

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

    # Vento
    tabela_vento = vento_ajustado.to_dataframe().reset_index()
    tabela_vento = tabela_vento.rename(columns={"wind_speed": "vento_velocidade"})
    tabela_vento["ano"] = tabela_vento["time"].dt.year
    tabela_vento["mes"] = tabela_vento["time"].dt.month
    tabela_vento = tabela_vento[["latitude", "longitude", "ano", "mes", "vento_velocidade"]]

    # Correntes: calcular velocidade total a partir de uo (leste) e vo (norte)
    tabela_correntes = correntes_ajustadas.to_dataframe().reset_index()
    tabela_correntes["corrente_velocidade"] = np.sqrt(tabela_correntes["uo"]**2 + tabela_correntes["vo"]**2)
    tabela_correntes["ano"] = tabela_correntes["time"].dt.year
    tabela_correntes["mes"] = tabela_correntes["time"].dt.month
    tabela_correntes = tabela_correntes[["latitude", "longitude", "ano", "mes", "corrente_velocidade"]]

    mensal = mensal.merge(tabela_vento, on=["latitude", "longitude", "ano", "mes"], how="left")
    mensal = mensal.merge(tabela_correntes, on=["latitude", "longitude", "ano", "mes"], how="left")

    tabelas_mensais.append(mensal)
    print(f"Ano {ano} processado: {len(mensal)} linhas mensais")

tabela_final = pd.concat(tabelas_mensais, ignore_index=True)

print(f"\nTotal de linhas: {len(tabela_final)}")
print(f"\nValores faltando:")
print(tabela_final.isna().sum())

tabela_final = tabela_final.dropna()
print(f"\nLinhas finais: {len(tabela_final)}")

tabela_final.to_csv("dados/dados_mensais_2000_2024_sp_completo.csv", index=False)
print("\nArquivo salvo: dados/dados_mensais_2000_2024_sp_completo.csv")
print("\nPrimeiras linhas:")
print(tabela_final.head(10))