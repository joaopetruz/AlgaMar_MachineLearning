"""Previsão dos próximos 12 meses a partir do último mês observado.

Saída padrão (uma linha por ponto × mês-alvo), colunas:
| campo                  | significado |
|------------------------|-------------|
| prediction_date        | data (UTC) em que a previsão foi gerada |
| issue_month            | mês de emissão t0 = último mês com estado ambiental observado (AAAA-MM) |
| target_date            | mês previsto t0+h (AAAA-MM) |
| forecast_horizon       | h em meses (1..12) |
| point_id, latitude, longitude | ponto da grade CHL de 4 km (WGS84) |
| region                 | região do config (provisória, ver DÚVIDA 2) |
| bloom_probability      | probabilidade estimada de clorofila máxima do mês > limiar do alvo |
| climatological_probability | frequência histórica de floração naquele ponto/mês (referência) |
| risk_level             | BAIXO / MODERADO / ALTO (limiares no cartão do modelo) |
| model_version          | versão do modelo |
| feature_flags          | grupos de features usados |
| data_sources           | produto usado para cada variável no mês de emissão |
Campos que NÃO existem nos dados e não são preenchidos: município, estação de coleta, espécie de alga, toxicidade.
"""
import datetime as dt

import numpy as np
import pandas as pd

from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import constroi_linhas, texto_mes
from algamar_ml.modelos.treino import probabilidade


def nivel_risco(prob, limiar_alto, limiar_moderado):
    return np.where(prob >= limiar_alto, "ALTO", np.where(prob >= limiar_moderado, "MODERADO", "BAIXO"))


def mes_de_emissao(cubo, cobertura_minima=0.9):
    """Último mês em que clorofila e temperatura existem em >= `cobertura_minima` dos pontos."""
    chl, sst = cubo.dados["clorofila_media"], cubo.dados["temperatura_media"]
    cobre = (np.mean(~np.isnan(chl), axis=1) >= cobertura_minima) & (np.mean(~np.isnan(sst), axis=1) >= cobertura_minima)
    if not cobre.any():
        raise ValueError("nenhum mês com estado ambiental suficiente para emitir previsão")
    return cubo.t_inicio + int(np.where(cobre)[0].max())


def preve(modelo, cartao, cubo, cfg, fisica=None, t0=None, fontes=None):
    t0 = mes_de_emissao(cubo) if t0 is None else t0
    linhas = constroi_linhas(cubo, cfg["alvo"]["limiar"], [t0], horizontes=cfg["previsao"]["horizontes"],
                             fisica=fisica, exige_estado=False)
    prob = probabilidade(modelo, linhas, cartao)
    limiar = cartao["limiar_alto"]
    saida = pd.DataFrame({
        "prediction_date": dt.datetime.now(dt.UTC).date().isoformat(),
        "issue_month": texto_mes(linhas["t0"].to_numpy()),
        "target_date": texto_mes(linhas["tau"].to_numpy()),
        "forecast_horizon": linhas["h"].astype(int),
        "point_id": linhas["point_id"].astype(int),
        "latitude": linhas["latitude"].astype(float).round(5),
        "longitude": linhas["longitude"].astype(float).round(5),
        "region": regiao(linhas["longitude"], cfg["regioes"]),
        "bloom_probability": prob.round(4),
        "climatological_probability": linhas["freq_floracao_alvo"].astype(float).round(4),
        "risk_level": nivel_risco(prob, limiar, cartao["limiar_moderado"]),
        "model_version": cartao["versao"],
        "feature_flags": ";".join(f"{k}={v}" for k, v in cartao["flags"].items()),
        "data_sources": fontes or "",
        "estado_t0_completo": linhas["chl_media_t0"].notna() & linhas["sst_t0"].notna(),
    })
    return saida


def resumo_regioes(previsao):
    """Risco por região e mês: probabilidade média e máxima, % de pontos em ALTO."""
    g = previsao.groupby(["region", "target_date", "forecast_horizon"])
    return g.agg(prob_media=("bloom_probability", "mean"), prob_max=("bloom_probability", "max"),
                 prob_climatologica_media=("climatological_probability", "mean"),
                 pontos=("point_id", "count"),
                 frac_pontos_alto=("risk_level", lambda s: float((s == "ALTO").mean()))).reset_index()


def pico_por_ponto(previsao):
    """Mês de maior probabilidade em cada ponto (quando) e o valor."""
    i = previsao.groupby("point_id")["bloom_probability"].idxmax()
    return previsao.loc[i, ["point_id", "latitude", "longitude", "region", "target_date", "forecast_horizon",
                            "bloom_probability", "risk_level"]].rename(columns={"target_date": "mes_pico"}).reset_index(drop=True)
