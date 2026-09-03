import xarray as xr

# Abrir o arquivo baixado
dados = xr.open_dataset("dados/teste_clorofila.nc")

# Mostrar um resumo do que tem dentro
print(dados)

# Converter para tabela (DataFrame)
tabela = dados.to_dataframe().reset_index()

# Mostrar as primeiras linhas
print(tabela.head(10))

# Mostrar quantas linhas no total
print(f"\nTotal de linhas: {len(tabela)}")

# Ver estatísticas básicas da clorofila
print(f"\nEstatísticas de CHL:")
print(tabela['CHL'].describe())

import matplotlib.pyplot as plt

# Pegar só o primeiro dia para visualizar
dia1 = dados["CHL"].isel(time=0)

# Plotar como mapa de calor
plt.figure(figsize=(8, 6))
dia1.plot(cmap="YlGn", vmin=0, vmax=2)
plt.title("Concentração de Clorofila-a — 01/01/2024")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("mapa_clorofila.png", dpi=150)
print("Mapa salvo como mapa_clorofila.png")
plt.show()