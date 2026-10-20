"""Endpoints da previsão dos próximos 12 meses.

A API não recalcula nada. Ela serve, nesta ordem:
1. a última execução do pipeline oficial (`algamar.py previsao`), apontada por saidas/ULTIMA.json
   (ou a mais recente em saidas/ com previsao.parquet, se a última não gerou probabilidades);
2. senão, a última previsão legada de prever_12_meses.py em previsoes/<mês de emissão>/.
As colunas do pipeline (contrato/saida_previsao_v1.json) são renomeadas para os nomes que a API já expunha.
"""
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from algamar_ml.espacial import haversine_km

_RAIZ = Path(__file__).resolve().parent.parent
PASTA = Path(os.environ.get("ALGAMAR_PREVISOES", _RAIZ / "previsoes"))
SAIDAS = Path(os.environ.get("ALGAMAR_SAIDAS", _RAIZ / "saidas"))
DIST_MAX_KM = float(os.environ.get("ALGAMAR_DIST_MAX_KM", "15"))

router = APIRouter(prefix="/forecast", tags=["previsão 12 meses"])

# pipeline oficial (fluxo.py) → nomes da API
_COLS_PIPELINE = {"horizon": "forecast_horizon", "regiao": "region", "probabilidade": "bloom_probability",
                  "probabilidade_climatologica": "climatological_probability", "nivel_de_risco": "risk_level",
                  "versao": "model_version"}
_COLS_REGIOES = {"regiao": "region", "horizon": "forecast_horizon", "frac_alto": "frac_pontos_alto"}


def _ultima_saida():
    """Pasta da execução oficial a servir: a de ULTIMA.json se tem probabilidades, senão a mais recente que tem."""
    if not SAIDAS.is_dir():
        return None
    ultima = SAIDAS / "ULTIMA.json"
    if ultima.exists():
        run_id = json.loads(ultima.read_text(encoding="utf-8")).get("run_id", "")
        if run_id and (SAIDAS / run_id / "previsao.parquet").exists():
            return SAIDAS / run_id
    pastas = sorted(p for p in SAIDAS.iterdir() if (p / "previsao.parquet").exists())
    return pastas[-1] if pastas else None


def _ultima_pasta_legada():
    pastas = sorted(p for p in PASTA.glob("*-*") if (p / "previsao_pontos.parquet").exists() and (p / "metadados.json").exists())
    return pastas[-1] if pastas else None


@lru_cache(maxsize=4)
def _carrega_saida(pasta: str, mtime: float):
    p = Path(pasta)
    prev = pd.read_parquet(p / "previsao.parquet").rename(columns=_COLS_PIPELINE)
    regioes = pd.read_parquet(p / "risco_regioes.parquet").rename(columns=_COLS_REGIOES)
    j = json.loads((p / "previsao.json").read_text(encoding="utf-8"))
    produto = j.get("produto") or {}
    resumo = j.get("resumo") or {}
    meta = {"issue_month": (j.get("periodo") or {}).get("emissao_t0"),
            "meses_previstos": sorted(prev["target_date"].unique().tolist()),
            "model_version": j.get("modelo"),
            "status_modelo": j.get("classificacao_produto"),
            "status_execucao": j.get("status"),
            "run_id": j.get("run_id"), "gerado_em": j.get("gerado_em"),
            "produto": produto.get("nome"), "descricao": produto.get("descricao"),
            "fontes_mes_emissao": "; ".join(produto.get("fontes_de_dados") or []),
            "pontos": resumo.get("pontos"), "pontos_com_estado_incompleto": resumo.get("pontos_insuficientes"),
            "origem": f"saidas/{p.name}"}
    return prev, meta, regioes


@lru_cache(maxsize=4)
def _carrega(pasta: str, mtime: float):
    p = Path(pasta)
    prev = pd.read_parquet(p / "previsao_pontos.parquet")
    meta = json.loads((p / "metadados.json").read_text(encoding="utf-8"))
    meta.setdefault("origem", f"previsoes/{p.name}")
    regioes = pd.read_csv(p / "resumo_regioes.csv")
    return prev, meta, regioes


def dados():
    saida = _ultima_saida()
    if saida is not None:
        return _carrega_saida(str(saida), (saida / "previsao.parquet").stat().st_mtime)
    pasta = _ultima_pasta_legada()
    if pasta is None:
        raise HTTPException(503, "Nenhuma previsão gerada ainda. Rode: python algamar.py previsao")
    return _carrega(str(pasta), (pasta / "previsao_pontos.parquet").stat().st_mtime)


def _limpa(registros):
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in registros]


COLUNAS_PONTO = ["target_date", "forecast_horizon", "bloom_probability", "climatological_probability", "risk_level"]


@router.get("/meta")
def meta():
    """Mês de emissão, meses previstos, versão e status do modelo, fontes dos dados."""
    return dados()[1]


@router.get("")
def previsao_no_local(latitude: float = Query(..., ge=-90, le=90), longitude: float = Query(..., ge=-180, le=180)):
    """Os 12 meses previstos no ponto da grade mais próximo (distância geodésica)."""
    prev, meta_, _ = dados()
    pontos = prev[["point_id", "latitude", "longitude", "region"]].drop_duplicates("point_id")
    d = haversine_km(latitude, longitude, pontos["latitude"].to_numpy(), pontos["longitude"].to_numpy())
    i = int(np.argmin(d))
    if d[i] > DIST_MAX_KM:
        raise HTTPException(404, f"Sem ponto de previsão a menos de {DIST_MAX_KM} km; o mais próximo está a {d[i]:.1f} km "
                                 f"({pontos['latitude'].iloc[i]:.4f}, {pontos['longitude'].iloc[i]:.4f}).")
    ponto = pontos.iloc[i]
    serie = prev[prev["point_id"] == ponto["point_id"]].sort_values("forecast_horizon")
    # ponto sem estado em t0: probabilidades nulas (o pipeline não inventa previsão)
    pico = serie.loc[serie["bloom_probability"].idxmax()] if serie["bloom_probability"].notna().any() else None
    return {"issue_month": meta_["issue_month"], "model_version": meta_["model_version"],
            "point_id": int(ponto["point_id"]), "latitude": float(ponto["latitude"]), "longitude": float(ponto["longitude"]),
            "region": ponto["region"], "distancia_km": round(float(d[i]), 2),
            "mes_de_maior_risco": None if pico is None else pico["target_date"],
            "probabilidade_maxima": None if pico is None else float(pico["bloom_probability"]),
            "meses": _limpa(serie[COLUNAS_PONTO].to_dict("records"))}


@router.get("/regioes")
def por_regiao():
    """Risco por região e mês (probabilidade média/máxima, fração de pontos em ALTO)."""
    _, meta_, regioes = dados()
    return {"issue_month": meta_["issue_month"], "model_version": meta_["model_version"],
            "regioes": _limpa(regioes.to_dict("records"))}


@router.get("/mapa")
def mapa(target_date: str = Query(..., pattern=r"^\d{4}-\d{2}$"), nivel_minimo: str = Query("BAIXO", pattern="^(BAIXO|MODERADO|ALTO)$")):
    """Todos os pontos de um mês previsto (onde), filtrando pelo nível mínimo de risco."""
    prev, meta_, _ = dados()
    ordem = {"BAIXO": 0, "MODERADO": 1, "ALTO": 2}
    sel = prev[(prev["target_date"] == target_date) & (prev["risk_level"].map(ordem) >= ordem[nivel_minimo])]
    if sel.empty and target_date not in set(prev["target_date"]):
        raise HTTPException(404, f"{target_date} não está entre os meses previstos: {sorted(prev['target_date'].unique())}")
    cols = ["point_id", "latitude", "longitude", "region", "bloom_probability", "climatological_probability", "risk_level"]
    return {"issue_month": meta_["issue_month"], "target_date": target_date, "pontos": _limpa(sel[cols].to_dict("records"))}
