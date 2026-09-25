import pandas as pd

precipitacao = pd.read_csv("dados/precipitacao_2000_2024_sp.csv")
base_salinidade = pd.read_csv("dados/dados_features_sazonal_sp_com_salinidade.csv")

print("Primeiras 5 coordenadas da PRECIPITAÇÃO:")
print(precipitacao[["latitude", "longitude"]].drop_duplicates().head(5).to_string(index=False))

print("\nPrimeiras 5 coordenadas da base de SALINIDADE (nossa grade principal):")
print(base_salinidade[["latitude", "longitude"]].drop_duplicates().head(5).to_string(index=False))

# Testar se existe ALGUMA coincidência exata
coords_precip = set(zip(precipitacao["latitude"].round(4), precipitacao["longitude"].round(4)))
coords_base = set(zip(base_salinidade["latitude"].round(4), base_salinidade["longitude"].round(4)))

interseccao = coords_precip & coords_base
print(f"\nCoordenadas em comum (arredondado 4 casas): {len(interseccao)}")
print(f"Total precipitação: {len(coords_precip)}")
print(f"Total base salinidade: {len(coords_base)}")