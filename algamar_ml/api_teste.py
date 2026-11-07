"""Teste da previsão no passado (hindcast): o modelo prevê a partir de um mês já observado e a resposta é
comparada com o que aconteceu de verdade. Página: GET /testar (testar_ml.html).

Sem vazamento por construção: o painel é CORTADO em t0 antes de montar as features, então o modelo não
tem como ver nada depois do mês de emissão (é o que uma previsão real feita em t0 teria visto). O valor
real (floração observada) vem do painel completo e só entra na comparação.

Aviso honesto: emissões cujos alvos caem até o fim da validação do modelo (cartão: split.fim_validacao)
são meses que ele já viu no treino/validação; só o período de teste do cartão em diante é previsão "às cegas".
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from algamar_ml import config, metricas, registro
from algamar_ml.api_dashboard import regiao_dashboard
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import alvo, constroi_linhas, cubo_do_painel, texto_mes
from algamar_ml.modelos.treino import probabilidade

PAGINA = Path(__file__).resolve().parent.parent / "testar_ml.html"
TODO_LITORAL = "Litoral todo"

router = APIRouter(tags=["teste do ML"])


def _mes(texto):
    try:
        ano, mes = (int(x) for x in texto.split("-"))
        if not 1 <= mes <= 12:
            raise ValueError
    except ValueError:
        raise HTTPException(422, f"mês inválido '{texto}' (use AAAA-MM)")
    return ano * 12 + mes - 1


@lru_cache(maxsize=1)
def _painel(caminho: str, mtime: float):
    painel = pd.read_parquet(caminho)
    painel["_t"] = painel["ano"] * 12 + painel["mes"] - 1
    return painel


@lru_cache(maxsize=1)
def _real(caminho: str, mtime: float, limiar: float):
    """Floração observada [mês, ponto] do painel completo e o índice (lat, lon) → coluna."""
    cubo = cubo_do_painel(_painel(caminho, mtime).drop(columns="_t"))
    idx = {(round(float(a), 5), round(float(o), 5)): int(i)
           for i, a, o in cubo.pontos[["point_id", "latitude", "longitude"]].itertuples(index=False)}
    return alvo(cubo, limiar), cubo.t_inicio, cubo.t_fim, idx


@lru_cache(maxsize=3)
def _modelo(raiz: str, versao: str):
    modelo, cartao, _ = registro.carrega(raiz, versao)
    return modelo, cartao


def _contexto(versao=None):
    cfg = config.carrega()
    caminho = cfg["caminhos"]["painel"]
    if not caminho.exists():
        raise HTTPException(503, f"Painel mensal não encontrado em {caminho}. Rode: python construir_painel.py")
    versao = versao or cfg["modelo"]["oficial"]
    try:
        modelo, cartao = _modelo(str(cfg["caminhos"]["modelos"]), versao)
    except registro.ErroRegistro as e:
        raise HTTPException(404, str(e))
    return cfg, str(caminho), caminho.stat().st_mtime, modelo, cartao


def _visto_ate(cartao, cfg):
    return (cartao.get("split") or cfg["split"])["fim_validacao"]


@router.get("/teste/meses")
def meses(versao: str | None = None):
    """Meses que dá para testar e a partir de quando o teste é às cegas para o modelo."""
    cfg, caminho, mtime, _, cartao = _contexto(versao)
    t = _painel(caminho, mtime)["_t"]
    ultimo = int(t.max())
    visto = _visto_ate(cartao, cfg)
    return {"modelo": cartao["versao"], "primeira_emissao": str(texto_mes(np.array([int(t.min()) + 12]))[0]),
            "ultima_emissao": str(texto_mes(np.array([ultimo - 1]))[0]), "ultimo_mes_observado": str(texto_mes(np.array([ultimo]))[0]),
            "alvos_vistos_no_treino_ate": visto,
            "primeira_emissao_as_cegas": str(texto_mes(np.array([_mes(visto)]))[0]),
            "limiar_alto": cartao["limiar_alto"], "limiar_moderado": cartao.get("limiar_moderado")}


def _metricas(y, p, limiar, ok):
    m = metricas.basicas(y[ok], p[ok], limiar)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def hindcast(emissao: str, versao=None):
    cfg, caminho, mtime, modelo, cartao = _contexto(versao)
    t0 = _mes(emissao)
    painel = _painel(caminho, mtime)
    if t0 < int(painel["_t"].min()) + 12 or t0 >= int(painel["_t"].max()):
        raise HTTPException(422, f"emissão fora do intervalo testável ({meses(versao)['primeira_emissao']} a "
                                 f"{meses(versao)['ultima_emissao']})")
    limiar_alvo = float(cfg["alvo"]["limiar"])

    # 1. previsão só com o que existia até t0
    cubo = cubo_do_painel(painel[painel["_t"] <= t0].drop(columns="_t"))
    linhas = constroi_linhas(cubo, limiar_alvo, [t0], horizontes=cfg["previsao"]["horizontes"], exige_estado=False)
    falta = [f for f in cartao["features"] if f not in linhas]
    if falta:
        raise HTTPException(503, f"o teste não monta as features {falta} (modelo com física?)")
    p = probabilidade(modelo, linhas, cartao).astype(float)
    obrig = [f for f in cfg["pipeline"]["features_obrigatorias_t0"] if f in linhas]
    p[linhas[obrig].isna().any(axis=1).to_numpy()] = np.nan              # sem estado em t0 → sem probabilidade

    # 2. o que aconteceu de verdade (painel completo)
    real, t_ini, t_fim, idx = _real(caminho, mtime, limiar_alvo)
    col = np.array([idx.get((round(float(a), 5), round(float(o), 5)), -1)
                    for a, o in zip(linhas["latitude"], linhas["longitude"])])
    tau = linhas["tau"].to_numpy()
    ok = (col >= 0) & (tau <= t_fim)
    y = np.full(len(linhas), np.nan)
    y[ok] = real[tau[ok] - t_ini, col[ok]]

    lim_a, lim_m = float(cartao["limiar_alto"]), float(cartao.get("limiar_moderado", cartao["limiar_alto"] / 2))
    t = pd.DataFrame({"alvo": texto_mes(tau), "h": linhas["h"].astype(int).to_numpy(),
                      "region": [regiao_dashboard(r) for r in regiao(linhas["longitude"], cfg["regioes"])],
                      "prob": p, "clim": linhas["freq_floracao_alvo"].to_numpy(dtype=float), "real": y})
    t["alerta"] = t["prob"] >= lim_a

    def agrega(g):
        obs = g["real"].notna() & g["prob"].notna()
        alerta, real_ = g["alerta"] & obs, (g["real"] == 1) & obs
        return pd.Series({"prob_prevista": g["prob"].mean(), "prob_climatologica": g["clim"].mean(),
                          "fracao_real": g.loc[obs, "real"].mean() if obs.any() else np.nan,
                          "pontos": int(g["prob"].notna().sum()), "pontos_alerta": int((g["alerta"] & g["prob"].notna()).sum()),
                          "pontos_floracao_real": int(real_.sum()), "alertas_certos": int((alerta & real_).sum()),
                          "observado": bool(obs.any())})

    tabela = pd.concat([t.assign(region=TODO_LITORAL), t]).groupby(["region", "alvo", "h"]).apply(agrega, include_groups=False).reset_index()

    def _r(v, c=4):
        return None if pd.isna(v) else round(float(v), c)

    regioes = {}
    for reg, g in tabela.groupby("region", sort=False):
        regioes[reg] = [{"alvo": r.alvo, "h": int(r.h), "prob_prevista": _r(r.prob_prevista), "prob_climatologica": _r(r.prob_climatologica),
                         "fracao_real": _r(r.fracao_real), "pontos": int(r.pontos), "pontos_alerta": int(r.pontos_alerta),
                         "pontos_floracao_real": int(r.pontos_floracao_real), "alertas_certos": int(r.alertas_certos),
                         "observado": bool(r.observado)} for r in g.sort_values("h").itertuples()]

    # modelo e climatologia nos MESMOS pontos-mês: observados e com probabilidade do modelo
    yv, pv, cv = t["real"].to_numpy(), t["prob"].to_numpy(), t["clim"].to_numpy()
    comum = ~np.isnan(yv) & ~np.isnan(pv) & ~np.isnan(cv)
    visto = _visto_ate(cartao, cfg)
    alvos_vistos = int((t["alvo"] <= visto).groupby(t["h"]).any().sum())
    return {
        "emissao": str(texto_mes(np.array([t0]))[0]), "modelo": cartao["versao"], "limiar_alto": lim_a, "limiar_moderado": lim_m,
        "ultimo_mes_observado": str(texto_mes(np.array([t_fim]))[0]),
        "as_cegas": alvos_vistos == 0, "meses_ja_vistos_no_treino": alvos_vistos, "alvos_vistos_no_treino_ate": visto,
        "metricas": ({"modelo": _metricas(yv, pv, lim_a, comum), "climatologia": _metricas(yv, cv, lim_a, comum)}
                     if comum.any() else None),
        "regioes": regioes,
    }


@router.get("/teste/hindcast")
def teste_hindcast(emissao: str = Query(..., pattern=r"^\d{4}-\d{2}$", description="mês de emissão (último mês que o modelo vê)"),
                   versao: str | None = Query(None, description="versão do modelo (padrão: o oficial do config)")):
    """Previsão feita em `emissao` (só com dados até ele) × o que aconteceu nos 12 meses seguintes."""
    return hindcast(emissao, versao)


@router.get("/testar", response_class=HTMLResponse, include_in_schema=False)
def pagina():
    return HTMLResponse(PAGINA.read_text(encoding="utf-8"))
