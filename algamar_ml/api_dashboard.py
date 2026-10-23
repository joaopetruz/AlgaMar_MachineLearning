"""Resposta pronta para o dashboard (o backend repassa em GET /api/ml/coastal-risks).

Formato igual ao que o frontend já lê: {success, summary, trends, data}, com um registro por MUNICÍPIO × mês:
- region = nome do município (o frontend mostra "Floração em <region>"); macro_region = Litoral Norte,
  Baixada Santista ou Litoral Sul (é por ela que o filtro `region` funciona);
- latitude/longitude = ponto de mar em frente ao município; o registro agrega os pontos da grade a até
  `RAIO_KM` dele (faixa costeira; o mar aberto entra só no resumo e na tendência);
- meses passados (os últimos `MESES_OBSERVADOS` do painel): tipo "observado"; probability = fração dos
  pontos com floração observada (clorofila máxima > limiar do alvo); traz environmental_factors;
- meses futuros: tipo "previsao", da previsão oficial (a mesma do /forecast); probability = média dos pontos;
  environmental_factors = os do ÚLTIMO MÊS MEDIDO do município, com mes_referencia dizendo qual é;
- environmental_factors.ph: pH de superfície do modelo Copernicus (algamar_ml/ph.py; python atualizar_ph.py),
  só para exibir; fica ausente se a tabela não existir.
Níveis (risk_level), a ÚNICA fonte do nível para todas as telas (summary.niveis traz as faixas):
BAIXO < limiar_moderado ≤ MODERADO < limiar_alto ≤ ALTO < LIMIAR_MUITO_ALTO ≤ MUITO ALTO.
limiar_moderado e limiar_alto vêm do cartão do modelo (validação); MUITO ALTO = 0,80, onde no teste do modelo
oficial a floração observada passou de ~88 % (o modelo é calibrado: 0,8–0,9 → 88 %; 0,9–1,0 → 99 %).
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from algamar_ml import config, ph
from algamar_ml.api_previsao import dados as dados_previsao
from algamar_ml.espacial import haversine_km, regiao

MESES_OBSERVADOS = 12
RAIO_KM = 20
LIMIAR_MUITO_ALTO = 0.80
NIVEIS = ("BAIXO", "MODERADO", "ALTO", "MUITO ALTO")

# setores do ML → regiões do dashboard (Peruíbe–Mongaguá fazem parte da Baixada Santista)
REGIOES_DASHBOARD = {"norte": "Litoral Norte", "centro": "Baixada Santista", "sul": "Baixada Santista",
                     "extremo sul": "Litoral Sul"}
# municípios costeiros de SP: ponto de mar em frente à orla (a grade de 4 km tem ponto a ≤ 2,5 km de cada um)
MUNICIPIOS = [
    ("Ubatuba", "Litoral Norte", -23.47, -44.98), ("Caraguatatuba", "Litoral Norte", -23.66, -45.36),
    ("Ilhabela", "Litoral Norte", -23.85, -45.25), ("São Sebastião", "Litoral Norte", -23.86, -45.50),
    ("Bertioga", "Baixada Santista", -23.88, -46.10), ("Guarujá", "Baixada Santista", -24.03, -46.22),
    ("Santos", "Baixada Santista", -24.02, -46.33), ("Praia Grande", "Baixada Santista", -24.07, -46.45),
    ("Mongaguá", "Baixada Santista", -24.14, -46.61), ("Itanhaém", "Baixada Santista", -24.23, -46.78),
    ("Peruíbe", "Baixada Santista", -24.37, -46.98), ("Iguape", "Litoral Sul", -24.75, -47.40),
    ("Ilha Comprida", "Litoral Sul", -24.92, -47.62), ("Cananéia", "Litoral Sul", -25.10, -47.85),
]
ANCORA = {m[0]: m for m in MUNICIPIOS}

router = APIRouter(tags=["dashboard"])


def regiao_dashboard(nome_ml):
    return REGIOES_DASHBOARD.get(str(nome_ml).split(" (")[0], str(nome_ml))


def municipio(latitude, longitude):
    """Município costeiro mais próximo de cada ponto, ou None se nenhum estiver a até RAIO_KM."""
    lat, lon = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    d = np.stack([haversine_km(lat, lon, a, o) for _, _, a, o in MUNICIPIOS], axis=-1)
    nomes = np.array([m[0] for m in MUNICIPIOS], dtype=object)[d.argmin(axis=-1)]
    return np.where(d.min(axis=-1) <= RAIO_KM, nomes, None)


def nivel(prob, limiar_alto, limiar_moderado):
    if prob is None or np.isnan(prob):
        return None
    if prob >= LIMIAR_MUITO_ALTO:
        return "MUITO ALTO"
    return "ALTO" if prob >= limiar_alto else "MODERADO" if prob >= limiar_moderado else "BAIXO"


def nivel_do_filtro(texto):
    """Aceita os nomes que o frontend manda (MÉDIO, Muito alto, MUITO_ALTO…) e devolve o de NIVEIS."""
    import unicodedata
    t = unicodedata.normalize("NFD", str(texto)).encode("ascii", "ignore").decode().upper().replace("_", " ").strip()
    if t.startswith("MUIT"):
        return "MUITO ALTO"
    if t.startswith("ALT"):
        return "ALTO"
    if t.startswith("MED") or t.startswith("MOD"):
        return "MODERADO"
    if t.startswith("BAIX"):
        return "BAIXO"
    raise HTTPException(422, f"nível desconhecido '{texto}' (use {', '.join(NIVEIS)})")


def faixas(limiar_alto, limiar_moderado):
    cortes = [0.0, limiar_moderado, limiar_alto, LIMIAR_MUITO_ALTO, 1.0]
    return [{"nivel": n, "de": round(cortes[i], 2), "ate": round(cortes[i + 1], 2)} for i, n in enumerate(NIVEIS)]


def _num(v, casas):
    return None if v is None or pd.isna(v) else round(float(v), casas)


def _local(nome):
    _, macro, lat, lon = ANCORA[nome]
    return {"region": nome, "macro_region": macro, "latitude": lat, "longitude": lon}


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
    """Pontos da grade nos últimos meses do painel, com floração observada, macrorregião e município."""
    painel = pd.read_parquet(caminho)
    chave = painel["ano"] * 12 + painel["mes"] - 1
    painel = painel[chave > chave.max() - MESES_OBSERVADOS].copy()
    cfg_reg = {"setores": [{"nome": n, "lon_max": l} for n, l in setores]}
    painel["macro_region"] = [regiao_dashboard(r) for r in regiao(painel["longitude"], cfg_reg)]
    painel["municipio"] = municipio(painel["latitude"], painel["longitude"])
    painel["floracao"] = np.where(painel["clorofila_maxima"].isna(), np.nan, painel["clorofila_maxima"] > limiar_alvo)
    return painel


def observado():
    cfg = config.carrega()
    caminho = cfg["caminhos"]["painel"]
    if not caminho.exists():
        raise HTTPException(503, f"Painel mensal não encontrado em {caminho}. Rode: python construir_painel.py")
    setores = tuple((s["nome"], float(s["lon_max"])) for s in cfg["regioes"]["setores"])
    return _observado(str(caminho), caminho.stat().st_mtime, float(cfg["alvo"]["limiar"]), setores)


@lru_cache(maxsize=2)
def _ph(caminho: str, mtime: float):
    t = pd.read_parquet(caminho)
    return {(r.municipio, int(r.ano), int(r.mes)): float(r.ph) for r in t.itertuples()}


def ph_por_municipio():
    """{(municipio, ano, mes): pH}; vazio se a tabela ainda não foi gerada."""
    caminho = ph.caminho_tabela(config.carrega())
    return _ph(str(caminho), caminho.stat().st_mtime) if caminho.exists() else {}


def previsto(prev, lim_a):
    """Pontos da previsão oficial com probabilidade (sem dado suficiente ficam de fora), com macrorregião e município."""
    p = prev[prev["bloom_probability"].notna()].copy()
    p["macro_region"] = p["region"].map(regiao_dashboard)
    p["municipio"] = municipio(p["latitude"], p["longitude"])
    p["alto"] = p["bloom_probability"] >= lim_a
    return p


def _agrega_obs(obs, chave):
    return (obs.groupby(chave + ["ano", "mes"])
            .agg(probability=("floracao", "mean"), high_risk_points=("floracao", "sum"),
                 temperature_celsius=("temperatura_media", "mean"), salinity_psu=("salinidade_media", "mean"),
                 chlorophyll_mg_m3=("clorofila_media", "mean"), chlorophyll_max_mg_m3=("clorofila_maxima", "mean"))
            .reset_index())


def _agrega_fut(fut, chave):
    return (fut.groupby(chave + ["target_date", "forecast_horizon"])
            .agg(probability=("bloom_probability", "mean"), climatological_probability=("climatological_probability", "mean"),
                 high_risk_points=("alto", "sum"), pontos=("alto", "size"))
            .reset_index())


def monta(region=None, risk_level=None, month=None):
    prev, meta, _ = dados_previsao()
    lim_a, lim_m = limiares(meta["model_version"])
    obs, fut = observado(), previsto(prev, lim_a)
    obs_mun, fut_mun = obs[obs["municipio"].notna()], fut[fut["municipio"].notna()]
    if region:   # filtro do dashboard: macrorregião (ou o nome de um município)
        # registros dos municípios: pela macrorregião DO MUNICÍPIO (Peruíbe é Baixada mesmo com pontos no setor sul)
        do_mun = lambda m: (m["municipio"] == region) | (m["municipio"].map(lambda n: ANCORA[n][1]) == region)
        obs_mun, fut_mun = obs_mun[do_mun(obs_mun)], fut_mun[do_mun(fut_mun)]
        # resumo e tendência: pela macrorregião de cada ponto
        obs = obs[(obs["macro_region"] == region) | (obs["municipio"] == region)]
        fut = fut[(fut["macro_region"] == region) | (fut["municipio"] == region)]

    data, ultimo_ef, phs = [], {}, ph_por_municipio()
    for r in _agrega_obs(obs_mun, ["municipio"]).sort_values(["ano", "mes"]).itertuples():
        ef = {"temperature_celsius": _num(r.temperature_celsius, 2), "salinity_psu": _num(r.salinity_psu, 3),
              "chlorophyll_mg_m3": _num(r.chlorophyll_mg_m3, 4), "chlorophyll_max_mg_m3": _num(r.chlorophyll_max_mg_m3, 4),
              "ph": _num(phs.get((r.municipio, int(r.ano), int(r.mes))), 3), "mes_referencia": f"{r.ano}-{r.mes:02d}"}
        ultimo_ef[r.municipio] = ef
        data.append({"id": f"{r.municipio}-{r.ano}-{r.mes:02d}", **_local(r.municipio), "year": int(r.ano), "month": int(r.mes),
                     "probability": _num(r.probability, 4), "risk_level": nivel(r.probability, lim_a, lim_m),
                     "tipo": "observado", "high_risk_points": int(r.high_risk_points), "environmental_factors": ef})
    for r in _agrega_fut(fut_mun, ["municipio"]).itertuples():
        ano, mes = (int(x) for x in str(r.target_date).split("-")[:2])
        data.append({"id": f"{r.municipio}-{ano}-{mes:02d}", **_local(r.municipio), "year": ano, "month": mes,
                     "probability": _num(r.probability, 4), "risk_level": nivel(r.probability, lim_a, lim_m),
                     "tipo": "previsao", "forecast_horizon": int(r.forecast_horizon),
                     "climatological_probability": _num(r.climatological_probability, 4),
                     "high_risk_points": int(r.high_risk_points),
                     # futuro não tem medição: condições do último mês medido, com o mês em mes_referencia
                     "environmental_factors": dict(ultimo_ef[r.municipio]) if r.municipio in ultimo_ef else None})

    # resumo = próximo mês previsto (h=1), com todos os pontos (costa e mar aberto)
    h1 = fut[fut["forecast_horizon"] == 1]
    niveis = {d["risk_level"] for d in data if d["tipo"] == "previsao" and d.get("forecast_horizon") == 1}
    summary = {"total_monitored": int(len(h1)), "high_risk_count": int(h1["alto"].sum()),
               "general_status": ("ALERTA" if niveis & {"ALTO", "MUITO ALTO"} else "ATENÇÃO" if "MODERADO" in niveis
                                  else "NORMAL"),
               "niveis": faixas(lim_a, lim_m),
               "issue_month": meta["issue_month"], "target_month": h1["target_date"].min() if len(h1) else None,
               "municipios": len({d["region"] for d in data}),
               "model_version": meta["model_version"], "gerado_em": meta.get("gerado_em")}

    # tendência mensal (todos os pontos): meses observados + o próximo mês previsto
    trends = []
    for r in _agrega_obs(obs.assign(_todos=1), ["_todos"]).itertuples():
        trends.append({"year": int(r.ano), "month": int(r.mes), "avg_probability": _num(r.probability, 4),
                       "avg_temperature": _num(r.temperature_celsius, 2), "high_risk_points": int(r.high_risk_points),
                       "tipo": "observado"})
    for r in _agrega_fut(h1.assign(_todos=1), ["_todos"]).itertuples():
        ano, mes = (int(x) for x in str(r.target_date).split("-")[:2])
        trends.append({"year": ano, "month": mes, "avg_probability": _num(r.probability, 4), "avg_temperature": None,
                       "high_risk_points": int(r.high_risk_points), "tipo": "previsao"})

    if risk_level:
        alvo = nivel_do_filtro(risk_level)
        data = [d for d in data if d["risk_level"] == alvo]
    if month:
        data = [d for d in data if d["month"] == month]
    return {"success": True, "summary": summary, "trends": trends, "data": data}


@router.get("/coastal-risks")
def coastal_risks(region: str | None = Query(None, description="Litoral Norte, Baixada Santista, Litoral Sul ou um município"),
                  risk_level: str | None = Query(None, description="BAIXO, MODERADO (ou MÉDIO), ALTO ou MUITO ALTO"),
                  month: int | None = Query(None, ge=1, le=12)):
    """Mapa, riscos, indicadores e tendências no formato do dashboard (o backend repassa esta resposta)."""
    return monta(region, risk_level, month)
