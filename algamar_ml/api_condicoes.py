"""GET /condicoes-propicias: índice de condições propícias para espécies tóxicas (regras da literatura).

Serve o que indice_condicoes.py gravou em saidas/condicoes_propicias.json. Cada município traz, por espécie,
o mês mais recente com dados ("agora", observado) e os 3 meses seguintes ("proximos", previstos). O índice diz se
o mar está favorável à espécie, NÃO se vai haver floração. É uma camada separada do risco do ML (/coastal-risks).

GET /condicoes-propicias/historico: a série mensal já observada do mesmo índice (dados/condicoes/indice_mensal.parquet,
gravado por indice_condicoes.py), só dos municípios de SP — as zonas de validação (PR, SC) ficam de fora.
"""
import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from algamar_ml import config

router = APIRouter(tags=["dashboard"])


def arquivo():
    return Path(config.carrega()["caminhos"].get("saidas", "saidas")) / "condicoes_propicias.json"


def monta(especie=None, region=None):
    arq = arquivo()
    if not arq.exists():
        raise HTTPException(503, "Índice de condições ainda não gerado (rode python indice_condicoes.py).")
    d = json.loads(arq.read_text(encoding="utf-8"))
    linhas = d["municipios"]
    if especie:
        linhas = [m for m in linhas if m["especie"].lower() == especie.lower()]
    if region:
        r = region.lower()
        linhas = [m for m in linhas if r in (m["macro_region"].lower(), m["municipio"].lower())]
    return {"success": True, "gerado_em": d["gerado_em"], "aviso": d["aviso"], "cortes": d["cortes"],
            "especies": sorted({m["especie"] for m in d["municipios"]}), "data": linhas}


def arquivo_historico():
    return Path(config.carrega()["caminhos"]["painel"]).parent.parent / "condicoes" / "indice_mensal.parquet"


_cache_hist = {}   # mtime → tabela (o parquet só muda quando indice_condicoes.py roda de novo)


def _tabela_historico():
    arq = arquivo_historico()
    if not arq.exists():
        raise HTTPException(503, "Histórico do índice ainda não gerado (rode python indice_condicoes.py).")
    mtime = arq.stat().st_mtime
    if mtime not in _cache_hist:
        df = pd.read_parquet(arq, columns=["local", "grupo", "especie", "mes", "indice", "nivel"])
        _cache_hist.clear()
        _cache_hist[mtime] = df[df.grupo != "validação"].dropna(subset=["indice"])
    return _cache_hist[mtime]


def monta_historico(especie=None, region=None, meses=24):
    df = _tabela_historico()
    if especie:
        df = df[df.especie.str.lower() == especie.lower()]
    if region:
        r = region.lower()
        df = df[(df.grupo.str.lower() == r) | (df.local.str.lower() == r)]
    todos = sorted(df.mes.unique())
    janela = todos[-max(1, min(int(meses), 400)):]
    df = df[df.mes.isin(janela)]
    toxinas = {}
    arq = arquivo()
    if arq.exists():
        toxinas = {m["especie"]: m.get("toxina") for m in json.loads(arq.read_text(encoding="utf-8"))["municipios"]}
    data = [{"municipio": r.local, "macro_region": r.grupo, "especie": r.especie, "toxina": toxinas.get(r.especie),
             "mes": r.mes, "indice": round(float(r.indice), 3), "nivel": r.nivel}
            for r in df.itertuples(index=False)]
    return {"success": True, "meses": janela, "especies": sorted(_tabela_historico().especie.unique()), "data": data}


@router.get("/condicoes-propicias/historico")
def condicoes_propicias_historico(especie: str | None = Query(None), region: str | None = Query(None),
                                  meses: int = Query(24, ge=1, le=400, description="últimos N meses com dado")):
    """Índice de condições mês a mês (já observado), por município e espécie."""
    return monta_historico(especie, region, meses)


@router.get("/condicoes-propicias")
def condicoes_propicias(especie: str | None = Query(None, description="Dinophysis ou Pseudo-nitzschia"),
                        region: str | None = Query(None, description="Litoral Norte, Baixada Santista, Litoral Sul ou um município")):
    """Condições propícias por município e espécie: agora + 1 a 3 meses (o backend repassa esta resposta)."""
    return monta(especie, region)
