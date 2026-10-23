"""pH da água do mar na superfície, por município costeiro e mês (só para exibir no dashboard).

Fonte: modelo biogeoquímico global da Copernicus (GLOBAL_ANALYSISFORECAST_BGC_001_028, conjunto de
carbonatos, diário, 0,25° ≈ 25 km). É MODELO, não medição: não existe pH de satélite. O pH não entra no ML.

Cada município recebe a célula de oceano mais próxima do seu ponto de mar (até `DIST_MAX_KM`); a grade de
0,25° não resolve baías e estuários, então o valor representa o mar em frente, não a praia.
Tabela gerada: dados/processados/ph_mensal_municipios.parquet (municipio, ano, mes, ph, dist_km, fonte).
Atualizar: python atualizar_ph.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml.apis.copernicus import Produto, baixa_mes
from algamar_ml.espacial import haversine_km

PRODUTO = Produto("cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m", "ANFC", "ph")
DIST_MAX_KM = 40
NOME_TABELA = "ph_mensal_municipios.parquet"


def caminho_tabela(cfg):
    return Path(cfg["caminhos"]["painel"]).parent / NOME_TABELA


def baixa(cfg, meses):
    """Baixa os meses [(ano, mes), ...] (pula os que já existem). Devolve os arquivos."""
    return [baixa_mes(cfg, "ph", PRODUTO, ano, mes) for ano, mes in meses]


def mensal_do_arquivo(arquivo):
    """Média do mês na superfície: (lat, lon, ph) das células de oceano."""
    import xarray as xr

    with xr.open_dataset(arquivo) as ds:
        ph = ds["ph"]
        if "depth" in ph.dims:
            ph = ph.isel(depth=0)
        m = ph.mean("time", skipna=True).load()
    lat, lon = np.meshgrid(m["latitude"].values, m["longitude"].values, indexing="ij")
    v = m.values
    ok = ~np.isnan(v)
    return lat[ok], lon[ok], v[ok]


def por_municipio(arquivos, municipios):
    """Tabela municipio × mês com o pH da célula de oceano mais próxima do ponto de mar de cada município."""
    linhas = []
    for arq in arquivos:
        ano, mes = (int(x) for x in Path(arq).stem.split("_")[-1].split("-"))
        lat, lon, v = mensal_do_arquivo(arq)
        for nome, _, a, o in municipios:
            d = haversine_km(lat, lon, a, o)
            i = int(np.argmin(d))
            if d[i] <= DIST_MAX_KM:
                linhas.append({"municipio": nome, "ano": ano, "mes": mes, "ph": round(float(v[i]), 3),
                               "dist_km": round(float(d[i]), 1), "fonte": PRODUTO.dataset_id})
    return pd.DataFrame(linhas)
