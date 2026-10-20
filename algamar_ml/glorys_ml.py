"""Física do GLORYS12 mensal (reanálise CMEMS) como feature do ML, no mês de emissão t0 (Etapa 13.7).

O mesmo dado e a mesma interpolação da 13.6 (etapa13_06_horizonte.py): cmems_mod_glo_phy_my_0.083deg_P1M-m, 1º nível
(0,49 m), interpolação bilinear da grade 1/12° para os pontos do ML. Meses sem GLORYS viram NaN; nada é preenchido.
Grupos:
  use_glorys_ts   → glorys_ts_t0 (thetao), glorys_ss_t0 (so)
  use_glorys_mld  → glorys_mld_t0 (mlotst)
"""
import numpy as np
import pandas as pd
import xarray as xr

from .features.horizonte import indice_mes

GRUPOS = {"use_glorys_ts": {"glorys_ts_t0": "thetao", "glorys_ss_t0": "so"},
          "use_glorys_mld": {"glorys_mld_t0": "mlotst"}}


def grupos_ativos(flags):
    return [g for g in GRUPOS if flags.get(g, False)]


def para_cubo(caminho, cubo, flags):
    """→ {nome: matriz [T, P] float32} alinhada ao cubo (mês de emissão)."""
    pedidas = {n: v for g in grupos_ativos(flags) for n, v in GRUPOS[g].items()}
    if not pedidas:
        return {}
    ds = xr.open_dataset(caminho)
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)
    la = xr.DataArray(cubo.pontos.latitude.to_numpy(), dims="p")
    lo = xr.DataArray(cubo.pontos.longitude.to_numpy(), dims="p")
    it = ds[list(set(pedidas.values()))].interp(latitude=la, longitude=lo)
    ti = np.array([indice_mes(t.year, t.month) for t in pd.to_datetime(it.time.values)]) - cubo.t_inicio
    ok = (ti >= 0) & (ti < cubo.T)
    out = {}
    for nome, var in pedidas.items():
        a = np.full((cubo.T, len(cubo.pontos)), np.nan, dtype=np.float32)
        a[ti[ok]] = it[var].values[ok]
        out[nome] = a
    return out
