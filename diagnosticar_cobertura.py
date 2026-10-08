import numpy as np
import pandas as pd
import xarray as xr

SANTOS = (-23.96, -46.35)


def km(lat, lon, ref=SANTOS):
    return np.hypot((lon - ref[1]) * 111 * np.cos(np.radians(ref[0])), (lat - ref[0]) * 111)


# ===== 1. Base final usada pela API =====
import sys

caminho_csv = sys.argv[1] if len(sys.argv) > 1 else "dados/dados_features_sazonal_sp_com_salinidade.csv"

base = pd.read_csv(caminho_csv, usecols=["latitude", "longitude"]).drop_duplicates()

print(f"=== Base final: {caminho_csv} ===")
print(f"Pontos únicos: {len(base)}")
print(f"Latitude:  {base.latitude.min():.3f} a {base.latitude.max():.3f}")
print(f"Longitude: {base.longitude.min():.3f} a {base.longitude.max():.3f}")

base["km_santos"] = km(base.latitude, base.longitude)
print("\n5 pontos mais próximos de Santos:")
print(base.sort_values("km_santos").head(5).to_string(index=False))

# ===== 2. Dados brutos de 2024: quem remove a costa? =====
print("\n=== Dados brutos 2024 ===")
chl = xr.open_dataset("dados/clorofila_2024_sp.nc")
sal = xr.open_dataset("dados/salinidade_2024_sp.nc")
sal_i = sal.interp(latitude=chl.latitude, longitude=chl.longitude)

def valido(da):
    """True onde há algum dado em qualquer tempo/profundidade, no formato (lat, lon)."""
    extras = [d for d in da.dims if d not in ("latitude", "longitude")]
    return da.notnull().any(extras).transpose("latitude", "longitude").values


chl_ok = valido(chl["CHL"])
sal_ok = valido(sal_i["sos"])
lat2d, lon2d = np.meshgrid(chl.latitude.values, chl.longitude.values, indexing="ij")
dist = km(lat2d, lon2d)

for nome, mascara in [
    ("Clorofila válida", chl_ok),
    ("Salinidade válida (interpolada)", sal_ok),
    ("Clorofila E salinidade válidas", chl_ok & sal_ok),
]:
    d = dist[mascara].min() if mascara.any() else float("nan")
    print(f"{nome}: {int(mascara.sum())} pontos; mais próximo de Santos a {d:.1f} km")

print(
    "\nSe 'Clorofila válida' chega perto de Santos mas 'Clorofila E salinidade' não, "
    "o dropna() da salinidade está removendo a costa."
)
