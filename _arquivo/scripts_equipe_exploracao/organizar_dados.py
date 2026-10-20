import xarray as xr

clorofila = xr.open_dataset("dados/clorofila_2024_sp.nc")
temperatura = xr.open_dataset("dados/temperatura_2024_sp.nc")

temperatura_ajustada = temperatura.interp(
    latitude=clorofila.latitude,
    longitude=clorofila.longitude
)

dados_completos = xr.merge([clorofila, temperatura_ajustada])
dados_completos["analysed_sst"] = dados_completos["analysed_sst"] - 273.15

tabela = dados_completos.to_dataframe().reset_index()
tabela = tabela.rename(columns={
    "CHL": "clorofila",
    "analysed_sst": "temperatura_celsius"
})

print("Antes de tratar NaN:")
print(tabela.isna().sum())

tabela = tabela.sort_values(["latitude", "longitude", "time"])
tabela["clorofila"] = tabela.groupby(["latitude", "longitude"])["clorofila"].transform(
    lambda x: x.interpolate(method="linear", limit_direction="both")
)
tabela["temperatura_celsius"] = tabela.groupby(["latitude", "longitude"])["temperatura_celsius"].transform(
    lambda x: x.interpolate(method="linear", limit_direction="both")
)

por_local = tabela.groupby(["latitude", "longitude"])["clorofila"].apply(lambda x: x.isna().mean())
locais_oceano = por_local[por_local < 1.0].index

tabela = tabela.set_index(["latitude", "longitude"])
tabela = tabela.loc[tabela.index.isin(locais_oceano)]
tabela = tabela.reset_index()

tabela = tabela.dropna(subset=["temperatura_celsius"])

print(f"\nLinhas finais (100% completas): {len(tabela)}")
print(tabela.isna().sum())

tabela.to_csv("dados/dados_tratados_2024_sp.csv", index=False)
print("\nArquivo salvo: dados/dados_tratados_2024_sp.csv")