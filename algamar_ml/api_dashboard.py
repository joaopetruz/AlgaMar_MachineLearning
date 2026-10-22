"""Resposta pronta para o dashboard (o backend repassa em GET /api/ml/coastal-risks).

Formato igual ao que o frontend já lê: {success, summary, trends, data}, com um registro por região × mês.
- meses passados (os últimos `MESES_OBSERVADOS` do painel): tipo "observado"; probability = fração dos
  pontos da região com floração observada (clorofila máxima > limiar do alvo); traz environmental_factors;
- meses futuros: tipo "previsao", da previsão oficial (a mesma do /forecast); probability = média dos pontos.
O nível da região usa os limiares do cartão do modelo, com os nomes do frontend (ALTO / MÉDIO / BAIXO).
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from algamar_ml import config
from algamar_ml.api_previsao import dados as dados_previsao
from algamar_ml.espacial import regiao

MESES_OBSERVADOS = 12

# setores do ML → regiões do dashboard (Peruíbe–Mongaguá fazem parte da Baixada Santista)
REGIOES_DASHBOARD = {"norte": "Litoral Norte", "centro": "Baixada Santista", "sul": "Baixada Santista",
                     "extremo sul": "Litoral Sul"}
# marcador de cada região no mapa: o ponto de mar que o frontend já usa (WEATHER_PLACES em dashboard.js);
# o centro dos pontos da grade cai ~100 km mar adentro
MARCADORES = {"Litoral Norte": (-23.5, -45.0), "Baixada Santista": (-24.04, -46.29), "Litoral Sul": (-25.1, -47.8)}

router = APIRouter(tags=["dashboard"])


def regiao_dashboard(nome_ml):
    return REGIOES_DASHBOARD.get(str(nome_ml).split(" (")[0], str(nome_ml))


def nivel(prob, limiar_alto, limiar_moderado):
    if prob is None or np.isnan(prob):
        return None
    return "ALTO" if prob >= limiar_alto else "MÉDIO" if prob >= limiar_moderado else "BAIXO"


def _num(v, casas):
    return None if v is None or pd.isna(v) else round(float(v), casas)


def _local(r):
    lat, lon = MARCADORES.get(r.region, (r.latitude, r.longitude))
    return {"latitude": _num(lat, 4), "longitude": _num(lon, 4)}


def limiares(model_version):
    cfg = config.carrega()
    caminho = cfg["caminhos"]["modelos"] / str(model_version) / "cartao.json"
    if not caminho.exists():
        raise HTTPException(503, f"Cartão do modelo não encontrado: {caminho}")
    cartao = json.loads(caminho.read_text(encoding="utf-8"))
    lim_a = float(cartao["limiar_alto"])
    lim_m = float(cartao.get("limiar_moderado", lim_a * cfg["previsao"].get("fracao_moderado", 0.5)))
    return lim_a, lim_m


@lru_cache(maxsize=2)
def _observado(caminho: str, mtime: float, limiar_alvo: float, setores: tuple) -> pd.DataFrame:
    """Agregado região × mês dos últimos meses do painel."""
    painel = pd.read_parquet(caminho)
    chave = painel["ano"] * 12 + painel["mes"] - 1
    painel = painel[chave > chave.max() - MESES_OBSERVADOS].copy()
    cfg_reg = {"setores": [{"nome": n, "lon_max": l} for n, l in setores]}
    painel["region"] = [regiao_dashboard(r) for r in regiao(painel["longitude"], cfg_reg)]
    painel["floracao"] = np.where(painel["clorofila_maxima"].isna(), np.nan, painel["clorofila_maxima"] > limiar_alvo)
    return (painel.groupby(["region", "ano", "mes"])
            .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"),
                 probability=("floracao", "mean"), high_risk_points=("floracao", "sum"),
                 temperature_celsius=("temperatura_media", "mean"), salinity_psu=("salinidade_media", "mean"),
                 chlorophyll_mg_m3=("clorofila_media", "mean"), chlorophyll_max_mg_m3=("clorofila_maxima", "mean"))
            .reset_index())


def observado():
    cfg = config.carrega()
    caminho = cfg["caminhos"]["painel"]
    if not caminho.exists():
        raise HTTPException(503, f"Painel mensal não encontrado em {caminho}. Rode: python construir_painel.py")
    setores = tuple((s["nome"], float(s["lon_max"])) for s in cfg["regioes"]["setores"])
    return _observado(str(caminho), caminho.stat().st_mtime, float(cfg["alvo"]["limiar"]), setores)


def previsto(prev, lim_a):
    """Agregado região × mês da previsão oficial (pontos sem dado suficiente ficam de fora)."""
    p = prev[prev["bloom_probability"].notna()].copy()
    p["region"] = p["region"].map(regiao_dashboard)
    p["alto"] = p["bloom_probability"] >= lim_a
    return (p.groupby(["region", "target_date", "forecast_horizon"])
            .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"),
                 probability=("bloom_probability", "mean"), climatological_probability=("climatological_probability", "mean"),
                 high_risk_points=("alto", "sum"), pontos=("alto", "size"))
            .reset_index())


def monta(region=None, risk_level=None, month=None):
    prev, meta, _ = dados_previsao()
    lim_a, lim_m = limiares(meta["model_version"])
    obs, fut = observado(), previsto(prev, lim_a)

    data = []
    for r in obs.itertuples():
        data.append({"id": f"{r.region}-{r.ano}-{r.mes:02d}", "region": r.region, "year": int(r.ano), "month": int(r.mes),
                     **_local(r),
                     "probability": _num(r.probability, 4), "risk_level": nivel(r.probability, lim_a, lim_m),
                     "tipo": "observado", "high_risk_points": int(r.high_risk_points),
                     "environmental_factors": {"temperature_celsius": _num(r.temperature_celsius, 2),
                                               "salinity_psu": _num(r.salinity_psu, 3),
                                               "chlorophyll_mg_m3": _num(r.chlorophyll_mg_m3, 4),
                                               "chlorophyll_max_mg_m3": _num(r.chlorophyll_max_mg_m3, 4)}})
    for r in fut.itertuples():
        ano, mes = (int(x) for x in str(r.target_date).split("-")[:2])
        data.append({"id": f"{r.region}-{ano}-{mes:02d}", "region": r.region, "year": ano, "month": mes,
                     **_local(r),
                     "probability": _num(r.probability, 4), "risk_level": nivel(r.probability, lim_a, lim_m),
                     "tipo": "previsao", "forecast_horizon": int(r.forecast_horizon),
                     "climatological_probability": _num(r.climatological_probability, 4),
                     "high_risk_points": int(r.high_risk_points), "environmental_factors": None})

    if region:
        data = [d for d in data if d["region"] == region]
        obs, fut = obs[obs["region"] == region], fut[fut["region"] == region]

    # resumo = próximo mês previsto (h=1)
    h1 = fut[fut["forecast_horizon"] == 1]
    niveis = {d["risk_level"] for d in data if d["tipo"] == "previsao" and d.get("forecast_horizon") == 1}
    summary = {"total_monitored": int(h1["pontos"].sum()), "high_risk_count": int(h1["high_risk_points"].sum()),
               "general_status": "ALERTA" if "ALTO" in niveis else "ATENÇÃO" if "MÉDIO" in niveis else "NORMAL",
               "issue_month": meta["issue_month"], "target_month": h1["target_date"].min() if len(h1) else None,
               "model_version": meta["model_version"], "gerado_em": meta.get("gerado_em")}

    # tendência mensal: meses observados + o próximo mês previsto
    trends = []
    for (ano, mes), g in obs.groupby(["ano", "mes"]):
        trends.append({"year": int(ano), "month": int(mes), "avg_probability": _num(g["probability"].mean(), 4),
                       "avg_temperature": _num(g["temperature_celsius"].mean(), 2),
                       "high_risk_points": int(g["high_risk_points"].sum()), "tipo": "observado"})
    for alvo, g in h1.groupby("target_date"):
        ano, mes = (int(x) for x in str(alvo).split("-")[:2])
        trends.append({"year": ano, "month": mes, "avg_probability": _num(np.average(g["probability"], weights=g["pontos"]), 4),
                       "avg_temperature": None, "high_risk_points": int(g["high_risk_points"].sum()), "tipo": "previsao"})

    if risk_level:
        data = [d for d in data if d["risk_level"] == risk_level]
    if month:
        data = [d for d in data if d["month"] == month]
    return {"success": True, "summary": summary, "trends": trends, "data": data}


@router.get("/coastal-risks")
def coastal_risks(region: str | None = Query(None, description="Litoral Norte, Baixada Santista ou Litoral Sul"),
                  risk_level: str | None = Query(None, pattern="^(ALTO|MÉDIO|BAIXO)$"),
                  month: int | None = Query(None, ge=1, le=12)):
    """Mapa, riscos, indicadores e tendências no formato do dashboard (o backend repassa esta resposta)."""
    return monta(region, risk_level, month)
