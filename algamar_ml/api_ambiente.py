"""GET /condicoes-ambientais: condições do mar mês a mês numa região (ou município), para a tela Previsões.

Uma linha por mês: os últimos `meses_passados` meses antes do mês atual e os meses da janela [hoje, hoje + meses]
(os mesmos da previsão). Cada variável vem com a fonte:
- "medido": o valor daquele mês (satélite no painel; correntes GLORYS; pH do modelo Copernicus);
- "normal": o mês ainda não tem medição; vai a média do mesmo mês do ano em 2000–2024 (NÃO é previsão);
- "ultimo": pH, que só existe a partir de 2025: o último mês disponível.
O local é a média dos municípios da região (pontos a até RAIO_KM de cada um). A corrente é o vetor médio do mês
(velocidade resultante e para onde vai). Vento e chuva NÃO vêm daqui: a média mensal das componentes do vento se anula
quando ele muda de direção (dá ~0 km/h); a tela pega os dois do histórico diário da Open-Meteo.
"""
import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import APIRouter, HTTPException, Query

from algamar_ml import config
from algamar_ml.api_dashboard import ANCORA, MUNICIPIOS, RAIO_KM, municipio, ph_por_municipio
from algamar_ml.api_previsao import meses_da_janela
from algamar_ml.espacial import haversine_km

NORMAL = (2000, 2024)
DIRS = ["N", "NE", "L", "SE", "S", "SO", "O", "NO"]
MS_KMH = 3.6

router = APIRouter(tags=["dashboard"])


def _vetor_local(da, lat, lon):
    """Série mensal (PeriodIndex) da média das células válidas a até RAIO_KM do ponto (ou da mais próxima)."""
    if "depth" in da.dims:
        da = da.isel(depth=0)
    a = da.sel(latitude=slice(lat - 1, lat + 1), longitude=slice(lon - 1, lon + 1))
    LA, LO = np.meshgrid(a.latitude.values, a.longitude.values, indexing="ij")
    d = haversine_km(lat, lon, LA, LO)
    valida = a.notnull().any("time").values
    sel = valida & (d <= RAIO_KM)
    if not sel.any():
        sel = np.where(valida, d, np.inf) == np.where(valida, d, np.inf).min()
    return pd.Series(np.nanmean(a.values[:, sel], axis=1), index=pd.DatetimeIndex(a.time.values).to_period("M"))


def _abre(nome):
    return xr.open_dataset(config.carrega()["caminhos"]["painel"].parent.parent / "condicoes" / f"{nome}.nc")


@lru_cache(maxsize=1)
def _tabela(painel: str, mtime: float) -> pd.DataFrame:
    """Uma linha por município × mês com todas as variáveis medidas."""
    p = pd.read_parquet(painel)
    pts = p[["latitude", "longitude"]].drop_duplicates()
    pts["municipio"] = municipio(pts["latitude"], pts["longitude"])
    p = p.merge(pts[pts["municipio"].notna()], on=["latitude", "longitude"])
    p["mes"] = pd.PeriodIndex.from_fields(year=p["ano"], month=p["mes"], freq="M")
    tab = (p.groupby(["municipio", "mes"])
           .agg(temperatura_c=("temperatura_media", "mean"), salinidade_psu=("salinidade_media", "mean"),
                clorofila_mg_m3=("clorofila_media", "mean"), clorofila_max_mg_m3=("clorofila_maxima", "mean")))

    glorys = _abre("glorys_mensal")
    extra = pd.concat([pd.DataFrame({"corrente_u": _vetor_local(glorys["uo"], lat, lon),
                                     "corrente_v": _vetor_local(glorys["vo"], lat, lon)})
                       .rename_axis("mes").assign(municipio=nome).reset_index()
                       for nome, _, lat, lon in MUNICIPIOS]).set_index(["municipio", "mes"])
    # pH por join externo: o modelo Copernicus tem o mês corrente, que o painel ainda não tem
    phs = pd.DataFrame([{"municipio": m, "mes": pd.Period(year=a, month=ms, freq="M"), "ph": v}
                        for (m, a, ms), v in ph_por_municipio().items()], columns=["municipio", "mes", "ph"])
    return tab.join(extra, how="outer").join(phs.set_index(["municipio", "mes"]), how="outer").reset_index()


def tabela():
    painel = config.carrega()["caminhos"]["painel"]
    if not painel.exists():
        raise HTTPException(503, f"Painel mensal não encontrado em {painel}. Rode: python construir_painel.py")
    return _tabela(str(painel), painel.stat().st_mtime)


def _direcao(u, v):
    """Para onde o vetor aponta (N, NE, L…)."""
    ang = np.degrees(np.arctan2(u, v)) % 360
    return DIRS[int((ang + 22.5) // 45) % 8]


ESCALARES = ["temperatura_c", "salinidade_psu", "clorofila_mg_m3", "clorofila_max_mg_m3", "ph"]
CASAS = {"temperatura_c": 1, "salinidade_psu": 2, "clorofila_mg_m3": 2, "clorofila_max_mg_m3": 2, "ph": 3}


def monta(region=None, meses_passados=9, meses=3, hoje=None):
    hoje = hoje or dt.date.today()
    if region and region not in ANCORA and region not in {m[1] for m in MUNICIPIOS}:
        raise HTTPException(422, f"região desconhecida '{region}' (use Litoral Norte, Baixada Santista, Litoral Sul ou um município)")
    locais = [m[0] for m in MUNICIPIOS if not region or region in (m[0], m[1])]
    t = tabela()
    t = t[t["municipio"].isin(locais)].groupby("mes").mean(numeric_only=True)
    anos = t.index.year
    normal = t[(anos >= NORMAL[0]) & (anos <= NORMAL[1])].groupby(lambda p: p.month).mean()
    ultimo_ph = t["ph"].dropna()

    atual = pd.Period(hoje, "M")
    pedidos = [atual - k for k in range(meses_passados, 0, -1)] + [pd.Period(m, "M") for m in meses_da_janela(hoje, meses)]
    linhas = []
    for mes in pedidos:
        med = t.loc[mes] if mes in t.index else None
        nor = normal.loc[mes.month] if mes.month in normal.index else None

        def valor(col):
            if med is not None and pd.notna(med[col]):
                return float(med[col]), "medido"
            if col == "ph":
                antes = ultimo_ph[ultimo_ph.index <= mes]
                return (float(antes.iloc[-1]), "ultimo") if len(antes) else (None, None)
            if nor is not None and pd.notna(nor[col]):
                return float(nor[col]), "normal"
            return None, None

        var = {}
        for col in ESCALARES:
            v, f = valor(col)
            var[col] = {"valor": None if v is None else round(v, CASAS[col]), "fonte": f}
        (u, f), (v, _) = valor("corrente_u"), valor("corrente_v")
        ok = u is not None and v is not None
        var["corrente_kmh"] = {"valor": round(float(np.hypot(u, v)) * MS_KMH, 2) if ok else None, "fonte": f if ok else None,
                               "direcao": _direcao(u, v) if ok else None}
        linhas.append({"mes": str(mes), "futuro": mes >= atual, "variaveis": var})
    return {"success": True, "region": region, "locais": locais, "normal": f"{NORMAL[0]}–{NORMAL[1]}",
            "hoje": hoje.isoformat(), "meses": linhas}


@router.get("/condicoes-ambientais")
def condicoes_ambientais(region: str | None = Query(None, description="Litoral Norte, Baixada Santista, Litoral Sul ou um município"),
                         meses_passados: int = Query(9, ge=0, le=24), meses: int = Query(3, ge=1, le=12)):
    """Temperatura, salinidade, clorofila, pH e corrente por mês: os passados medidos e os da janela da previsão."""
    return monta(region, meses_passados, meses)
