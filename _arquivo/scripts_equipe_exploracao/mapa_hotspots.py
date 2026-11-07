import pandas as pd
import matplotlib.pyplot as plt

tabela = pd.read_csv("dados/dados_tratados_2024_sp.csv", parse_dates=["time"])
media_por_local = tabela.groupby(["latitude", "longitude"])["clorofila"].mean().reset_index()

plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    media_por_local["longitude"], media_por_local["latitude"],
    c=media_por_local["clorofila"], cmap="YlOrRd", s=20, vmin=0, vmax=1.5
)
plt.colorbar(scatter, label="Clorofila média anual (mg/m³)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Mapa de Hotspots — Litoral SP — 2024")
plt.savefig("mapa_hotspots_sp.png", dpi=150)
print("Mapa salvo: mapa_hotspots_sp.png")

# Mostrar as 5 coordenadas com maior clorofila média (candidatas a hotspot)
top5 = media_por_local.sort_values("clorofila", ascending=False).head(5)
print("\nTop 5 pontos de maior concentração média:")
print(top5)

plt.show()