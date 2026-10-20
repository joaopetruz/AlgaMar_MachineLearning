"""Dados diários da Copernicus → tabela mensal por ponto da grade CHL (mesmo método da equipe).

Método (= processar_dados_historicos.py da equipe):
- SST e SSS são interpoladas (bilinear, xarray.interp) para a grade CHL de 4 km, dia a dia;
- SST: K → °C;
- por ponto e mês civil: clorofila média e máxima, temperatura média, salinidade média
  (médias ignoram dias sem dado).
"""
import numpy as np
import pandas as pd
import xarray as xr

CASAS = 5
NOMES = {"clorofila": "CHL", "temperatura": "analysed_sst", "salinidade": "sos"}


def chave_ponto(lat, lon):
    return np.round(np.asarray(lat, dtype=float), CASAS), np.round(np.asarray(lon, dtype=float), CASAS)


def _superficie(da):
    return da.isel(depth=0, drop=True) if "depth" in da.dims else da


def mensal_clorofila(ds):
    chl = _superficie(ds[NOMES["clorofila"]])
    mes = chl["time"].dt.strftime("%Y-%m")
    media = chl.groupby(mes).mean("time", skipna=True)
    maxima = chl.groupby(mes).max("time", skipna=True)
    return media, maxima


def mensal_interpolado(ds, variavel, grade_chl):
    da = _superficie(ds[NOMES[variavel]])
    da = da.interp(latitude=grade_chl["latitude"], longitude=grade_chl["longitude"])
    if variavel == "temperatura":
        da = da - 273.15
    mes = da["time"].dt.strftime("%Y-%m")
    return da.groupby(mes).mean("time", skipna=True)


def para_tabela(da, nome):
    tabela = da.rename(nome).to_dataframe().reset_index()
    col_mes = [c for c in tabela.columns if c not in ("latitude", "longitude", nome)][0]
    ano_mes = tabela.pop(col_mes).str.split("-", expand=True).astype(int)
    tabela["ano"], tabela["mes"] = ano_mes[0], ano_mes[1]
    tabela["latitude"], tabela["longitude"] = chave_ponto(tabela["latitude"], tabela["longitude"])
    return tabela


def processa(ds_chl, ds_sst=None, ds_sss=None):
    """Tabela mensal (latitude, longitude, ano, mes, clorofila_media, clorofila_maxima,
    temperatura_media, salinidade_media) na grade CHL. Variável ausente = NaN."""
    media, maxima = mensal_clorofila(ds_chl)
    tabela = para_tabela(media, "clorofila_media").merge(
        para_tabela(maxima, "clorofila_maxima"), on=["latitude", "longitude", "ano", "mes"])
    grade = ds_chl[["latitude", "longitude"]]
    for variavel, ds, nome in (("temperatura", ds_sst, "temperatura_media"), ("salinidade", ds_sss, "salinidade_media")):
        if ds is None:
            tabela[nome] = np.nan
            continue
        parte = para_tabela(mensal_interpolado(ds, variavel, grade), nome)
        tabela = tabela.merge(parte, on=["latitude", "longitude", "ano", "mes"], how="left")
    return tabela


def abre(caminhos):
    caminhos = [str(c) for c in caminhos]
    if not caminhos:
        return None
    return xr.open_mfdataset(caminhos, combine="by_coords") if len(caminhos) > 1 else xr.open_dataset(caminhos[0])
