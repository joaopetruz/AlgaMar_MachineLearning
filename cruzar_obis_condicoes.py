"""Condições ambientais (dados/condicoes) no mês e local de cada registro do OBIS (célula válida mais próxima, ≤ 30 km).
Uso: python cruzar_obis_condicoes.py dados  → dados/obis/obis_condicoes.csv"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

base = Path(sys.argv[1])
rows = []
for f in (base / "obis").glob("*.json"):
    for r in json.loads(f.read_text(encoding="utf-8")):
        r["especie"] = f.stem.replace("_", " "); rows.append(r)
d = pd.DataFrame(rows)
def mes_unico(e):
    if not isinstance(e, str): return None
    ms = [re.match(r"(\d{4})-(\d{1,2})", p) for p in e.split("/")]
    if not all(ms): return None
    v = {(int(m.group(1)), int(m.group(2))) for m in ms}
    if len(v) != 1: return None
    a, m = v.pop()
    return "%d-%02d" % (a, m) if 1 <= m <= 12 else None
d["mes"] = d.eventDate.map(mes_unico)
d = d[d.mes.notna() & (d.mes >= "2000-01")].drop_duplicates(["especie", "mes", "decimalLatitude", "decimalLongitude"]).reset_index(drop=True)

c = base / "condicoes"
g = xr.open_dataset(c / "glorys_mensal.nc").isel(depth=0)
ch = xr.open_dataset(c / "clorofila_grupos_mensal.nc")
w = xr.open_dataset(c / "vento_mensal.nc")
def km(lat1, lon1, lat2, lon2):
    return 111.2 * np.hypot(lat2 - lat1, (lon2 - lon1) * np.cos(np.radians(lat1)))
def pega(ds, var, lat, lon, mes, dmax=30):
    t = np.datetime64(mes + "-01")
    if t < ds.time.values.min() or t > ds.time.values.max(): return np.nan, np.nan
    a = ds[var].sel(time=t, method="nearest").sel(latitude=slice(lat - .4, lat + .4), longitude=slice(lon - .4, lon + .4))
    LA, LO = np.meshgrid(a.latitude.values, a.longitude.values, indexing="ij")
    dist = km(lat, lon, LA, LO); v = a.values; ok = ~np.isnan(v) & (dist <= dmax)
    if not ok.any(): return np.nan, np.nan
    i = np.argmin(np.where(ok, dist, np.inf)); iy, ix = np.unravel_index(i, v.shape)
    # percentil do valor entre todos os anos do mesmo mês do calendário na mesma célula
    serie = ds[var].sel(latitude=a.latitude.values[iy], longitude=a.longitude.values[ix])
    mesmo = serie.where(serie.time.dt.month == int(mes[5:7]), drop=True).values
    mesmo = mesmo[~np.isnan(mesmo)]
    pct = float((mesmo < v[iy, ix]).mean() * 100) if len(mesmo) > 3 else np.nan
    return float(v[iy, ix]), pct
out = []
for r in d.itertuples():
    lat, lon, m = r.decimalLatitude, r.decimalLongitude, r.mes
    rec = {"especie": r.especie, "mes": m, "lat": round(lat, 3), "lon": round(lon, 3)}
    for nome, ds, var in [("sst", g, "thetao"), ("sal", g, "so"), ("mld", g, "mlotst"), ("chl", ch, "CHL"), ("dino", ch, "DINO"),
                          ("diato", ch, "DIATO"), ("u_vento", w, "eastward_wind"), ("v_vento", w, "northward_wind")]:
        rec[nome], rec[nome + "_pct"] = pega(ds, var, lat, lon, m)
    out.append(rec)
t = pd.DataFrame(out)
t["frac_dino"] = t.dino / t.chl
t.to_csv(base / "obis" / "obis_condicoes.csv", index=False)
pd.set_option("display.width", 250)
print(t.groupby("especie").size().to_string())
print(t.groupby("especie")[["sst", "sal", "mld", "chl", "frac_dino", "u_vento", "v_vento"]].describe(percentiles=[.1, .5, .9]).T.round(2).to_string())
print("percentis (mesmo mês, mesma célula) - mediana:")
print(t.groupby("especie")[["sst_pct", "sal_pct", "mld_pct", "chl_pct", "dino_pct", "v_vento_pct"]].median().round(0).to_string())
print("registros sem SST:", int(t.sst.isna().sum()), " sem CHL:", int(t.chl.isna().sum()), " sem vento:", int(t.u_vento.isna().sum()))
