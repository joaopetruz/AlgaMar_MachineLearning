"""Localização: regiões, distância geodésica e associação ponto ↔ grade.

Métodos de associação (documentados na saída):
- "vizinho": ponto da grade mais próximo pela distância de haversine (km), recusado acima de
  `dist_max_km`. Não cria valores novos; é o método padrão para juntar grades diferentes.
- "bilinear": interpolação bilinear em grade regular lat/lon (xarray.interp), para campos
  contínuos; NaN onde algum vizinho é NaN (terra/fora do domínio).
"""
import numpy as np
import pandas as pd

RAIO_TERRA_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


def regiao(longitude, cfg_regioes):
    """Nome da região de cada ponto pelos setores de longitude do config (limite superior inclusivo)."""
    longitude = np.asarray(longitude, dtype=float)
    nomes = np.full(longitude.shape, None, dtype=object)
    for setor in reversed(cfg_regioes["setores"]):
        nomes[longitude <= setor["lon_max"]] = setor["nome"]
    return nomes


# Divisa SP/PR no mar (APROXIMAÇÃO, não é a linha legal do IBGE): reta a 135° (perpendicular à costa) partindo da
# Barra do Ararapira (−48,10; −25,36), extremo sul da costa paulista (IBGE 2025 × GSHHG, etapa 12.12 parte 4).
DIVISA_SP_PR = (-48.10, -25.36)


def dentro_sp(latitude, longitude, cfg_area):
    """Pontos do ML no litoral de SP: a caixa da equipe ([area].caixa_equipe) mais a extensão sul (lon < lon_min da
    caixa da equipe) dentro de [area] e do lado paulista da divisa SP/PR."""
    lat = np.asarray(latitude, dtype=float); lon = np.asarray(longitude, dtype=float)
    a = cfg_area
    caixa = (lat >= a["lat_min"]) & (lat <= a["lat_max"]) & (lon >= a["lon_min"]) & (lon <= a["lon_max"])
    e = a.get("caixa_equipe")
    if not e:
        return caixa
    equipe = (lon >= e[0]) & (lon <= e[1]) & (lat >= e[2]) & (lat <= e[3])
    lo0, la0 = DIVISA_SP_PR
    sul = caixa & (lon < e[0]) & (lon >= lo0) & (lat >= la0 - (lon - lo0))
    return equipe | sul


def vizinho_mais_proximo(lat_pontos, lon_pontos, lat_grade, lon_grade, dist_max_km):
    """Para cada ponto, o índice do ponto da grade mais próximo e a distância (km).
    Índice -1 quando a distância passa de `dist_max_km`."""
    lat_g = np.asarray(lat_grade, dtype=float)
    lon_g = np.asarray(lon_grade, dtype=float)
    indices, distancias = [], []
    for la, lo in zip(np.asarray(lat_pontos, dtype=float), np.asarray(lon_pontos, dtype=float)):
        d = haversine_km(la, lo, lat_g, lon_g)
        i = int(np.nanargmin(d))
        indices.append(i if d[i] <= dist_max_km else -1)
        distancias.append(float(d[i]))
    return np.array(indices), np.array(distancias)


def associa_tabela(pontos, fonte, colunas, dist_max_km, chave_tempo=("ano", "mes")):
    """Associa uma tabela de outra grade (`fonte`: latitude, longitude, ano, mes, colunas) aos
    `pontos` do ML pelo vizinho mais próximo. Devolve a tabela nos pontos do ML, com
    `dist_associacao_km` e `metodo_associacao`."""
    grade = fonte[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    idx, dist = vizinho_mais_proximo(pontos["latitude"], pontos["longitude"], grade["latitude"], grade["longitude"], dist_max_km)
    mapa = pontos[["latitude", "longitude"]].copy()
    mapa["lat_fonte"] = np.where(idx >= 0, grade["latitude"].to_numpy()[np.maximum(idx, 0)], np.nan)
    mapa["lon_fonte"] = np.where(idx >= 0, grade["longitude"].to_numpy()[np.maximum(idx, 0)], np.nan)
    mapa["dist_associacao_km"] = dist
    f = fonte.rename(columns={"latitude": "lat_fonte", "longitude": "lon_fonte"})
    saida = mapa.merge(f[["lat_fonte", "lon_fonte", *chave_tempo, *colunas]], on=["lat_fonte", "lon_fonte"], how="inner")
    saida["metodo_associacao"] = f"vizinho_mais_proximo(haversine, max {dist_max_km} km)"
    return saida.drop(columns=["lat_fonte", "lon_fonte"]), mapa


def bilinear(campo, lat_pontos, lon_pontos):
    """Interpolação bilinear de um xarray.DataArray (dims latitude/longitude) nos pontos."""
    import xarray as xr

    la = xr.DataArray(np.asarray(lat_pontos, dtype=float), dims="ponto")
    lo = xr.DataArray(np.asarray(lon_pontos, dtype=float), dims="ponto")
    return campo.interp(latitude=la, longitude=lo, method="linear")
