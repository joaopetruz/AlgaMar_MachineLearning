"""Alinhamento temporal (registros → mês civil UTC) e espacial (grade da fonte → pontos do ML)."""
import numpy as np
import pandas as pd

from algamar_ml.espacial import associa_tabela

COBERTURA_MINIMA = 0.8   # fração dos dias do mês com dado para aceitar a média mensal


def mensal(tabela, colunas, cobertura_minima=COBERTURA_MINIMA):
    """Registros com `time` (timestamp com fuso UTC) → média do mês civil UTC por ponto.

    Recusa tempo sem fuso (ambíguo) e registros repetidos no mesmo instante e ponto. Meses
    com cobertura de dias abaixo do mínimo ficam NaN (não são descartados em silêncio) e a
    coluna `cobertura_dias` registra a fração."""
    tempo = pd.to_datetime(tabela["time"])
    if tempo.dt.tz is None:
        raise ValueError("coluna time sem fuso horário: use UTC explícito (ex.: 2017-11-17T12:00Z)")
    tempo = tempo.dt.tz_convert("UTC")
    if tabela.assign(_t=tempo).duplicated(["latitude", "longitude", "_t"]).any():
        raise ValueError("registros repetidos (mesmo ponto e instante)")
    t = tabela.assign(ano=tempo.dt.year, mes=tempo.dt.month, dia=tempo.dt.day, _dias_mes=tempo.dt.days_in_month)
    g = t.groupby(["latitude", "longitude", "ano", "mes"])
    saida = g[colunas].mean()
    saida["cobertura_dias"] = g["dia"].nunique() / g["_dias_mes"].first()
    saida.loc[saida["cobertura_dias"] < cobertura_minima, colunas] = np.nan
    return saida.reset_index()


def para_pontos_ml(tabela_mensal, pontos_ml, colunas, dist_max_km):
    """Grade da fonte → pontos do ML pelo vizinho mais próximo (haversine, até dist_max_km)."""
    if _mesma_grade(tabela_mensal, pontos_ml):
        tabela_mensal = tabela_mensal.assign(latitude=tabela_mensal["latitude"].round(5),
                                             longitude=tabela_mensal["longitude"].round(5))
        pontos_ml = pontos_ml.assign(latitude=pontos_ml["latitude"].round(5), longitude=pontos_ml["longitude"].round(5))
        out = tabela_mensal.merge(pontos_ml[["latitude", "longitude"]], on=["latitude", "longitude"], how="inner")
        out["dist_associacao_km"] = 0.0
        out["metodo_associacao"] = "mesma_grade(round 5)"
        return out, None
    return associa_tabela(pontos_ml, tabela_mensal, colunas + [c for c in ("validade_espacial",) if c in tabela_mensal], dist_max_km)


def _mesma_grade(fonte, pontos):
    a = set(map(tuple, fonte[["latitude", "longitude"]].round(5).drop_duplicates().to_numpy()))
    b = set(map(tuple, pontos[["latitude", "longitude"]].round(5).to_numpy()))
    return len(b - a) <= 0.05 * len(b)
