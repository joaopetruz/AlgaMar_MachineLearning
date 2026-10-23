import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import create_model

from algamar_ml import config
from algamar_ml.api_dashboard import router as router_dashboard
from algamar_ml.api_previsao import dados as dados_previsao
from algamar_ml.api_previsao import router as router_previsao
from algamar_ml.api_teste import router as router_teste
from algamar_ml.features import climatologia_para_previsao

BASE_DIR = Path(__file__).resolve().parent

# Distância máxima (em graus, ~22 km) entre a coordenada pedida e um ponto da
# grade. Acima disso, a API devolve erro em vez de um ponto qualquer.
DIST_MAX_GRAUS = float(os.environ.get("ALGAMAR_DIST_MAX", "0.5"))

app = FastAPI(title="API de Previsão de Floração de Algas - AlgarMar (Litoral SP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir ao domínio do frontend em produção
    allow_methods=["*"],
    allow_headers=["*"],
)

# Previsão dos próximos 12 meses (nova): /forecast, /forecast/regioes, /forecast/mapa, /forecast/meta
app.include_router(router_previsao)
# Mesmo formato do GET /api/ml/coastal-risks do backend (o backend só repassa): /coastal-risks
app.include_router(router_dashboard)
# Teste da previsão no passado: página /testar, /teste/meses e /teste/hindcast?emissao=AAAA-MM
app.include_router(router_teste)


def carregar(nome: str):
    caminho = BASE_DIR / nome
    if not caminho.exists():
        raise RuntimeError(
            f"Arquivo '{nome}' não encontrado em {BASE_DIR}. "
            "Rode salvar_modelo_sazonal.py antes de iniciar a API."
        )
    return joblib.load(caminho)


def espalha(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Até `limit` linhas espaçadas ao longo da grade (ordenada por latitude/longitude),
    para cobrir a costa inteira e não só o extremo norte.
    """
    df = df.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    if limit >= len(df):
        return df
    posicoes = np.linspace(0, len(df) - 1, limit).round().astype(int)
    return df.iloc[posicoes]


@lru_cache(maxsize=2)
def _ultimo_mes_painel(caminho: str, mtime: float) -> pd.DataFrame:
    """Último mês do painel mensal atualizado (o mesmo que alimenta a previsão oficial)."""
    painel = pd.read_parquet(caminho)
    ano, mes = painel[["ano", "mes"]].sort_values(["ano", "mes"]).iloc[-1]
    return painel[(painel["ano"] == ano) & (painel["mes"] == mes)]


def ultimo_mes_painel() -> pd.DataFrame:
    caminho = config.carrega()["caminhos"]["painel"]
    if not caminho.exists():
        raise HTTPException(503, f"Painel mensal não encontrado em {caminho}. Rode: python construir_painel.py")
    return _ultimo_mes_painel(str(caminho), caminho.stat().st_mtime)


# Modelo sazonal antigo (classifica um mês já observado; NÃO é previsão futura).
# É opcional: sem os .pkl a API sobe só com os endpoints /forecast.
LEGADO_ERRO = None
try:
    modelo = carregar("modelo_floracao_sazonal.pkl")
    features = carregar("features_modelo_sazonal.pkl")
    limiar = carregar("limiar_decisao_sazonal.pkl")
    versao = carregar("versao_modelo.pkl")

    caminho_csv = BASE_DIR / "dados" / "dados_features_sazonal_sp_com_salinidade.csv"
    if not caminho_csv.exists():
        raise RuntimeError(f"Base de dados '{caminho_csv}' não encontrada.")

    # Ordena uma vez: "mais recentes" passa a ser realmente o fim da tabela
    dados_ambientais = (
        pd.read_csv(caminho_csv)
        .sort_values(["ano", "mes", "latitude", "longitude"])
        .reset_index(drop=True)
    )

    # Grade de pontos únicos e climatologia por ponto/mês, calculadas uma vez
    _pontos = dados_ambientais[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    _lats = _pontos["latitude"].to_numpy()
    _lons = _pontos["longitude"].to_numpy()

    # Para prever meses depois da base, todos os anos já são passado: climatologia com todos eles.
    # (A coluna do CSV é a climatologia "até o ano da linha", usada só no treino.)
    _clim = climatologia_para_previsao(dados_ambientais)

    # Schema do /predict gerado a partir das features do modelo salvo
    # (se a lista de features mudar, a API acompanha sozinha)
    DadosSazonais = create_model(
        "DadosSazonais",
        **{f: ((int if f == "mes" else float), ...) for f in features},
    )
except RuntimeError as erro:
    LEGADO_ERRO = str(erro)

    class DadosSazonais(create_model("DadosSazonais_vazio")):
        pass


def exige_legado():
    if LEGADO_ERRO:
        raise HTTPException(status_code=503, detail=f"Modelo sazonal antigo indisponível: {LEGADO_ERRO}")


def num(valor, casas):
    """Converte para float arredondado; NaN vira None (NaN quebra o JSON)."""
    valor = float(valor)
    return None if math.isnan(valor) else round(valor, casas)


@app.get("/")
def raiz():
    return {"mensagem": "API AlgarMar funcionando. Previsão dos próximos 12 meses em /forecast; /docs para testar.",
            "modelo_sazonal_antigo": "indisponível" if LEGADO_ERRO else "carregado"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/marine-data")
def marine_data(limit: int = Query(100, ge=1, le=5000)):
    """Dados ambientais do mês mais recente do painel atualizado, distribuídos por toda a costa."""
    amostra = espalha(ultimo_mes_painel(), limit)

    resultado = []
    for _, linha in amostra.iterrows():
        resultado.append({
            "latitude": num(linha["latitude"], 4),
            "longitude": num(linha["longitude"], 4),
            "year": int(linha["ano"]),
            "month": int(linha["mes"]),
            "temperature_celsius": num(linha["temperatura_media"], 2),
            "chlorophyll_mg_m3": num(linha["clorofila_media"], 4),
            "chlorophyll_max_mg_m3": num(linha["clorofila_maxima"], 4),
            "salinity_psu": num(linha["salinidade_media"], 3),
            "source": "Copernicus Marine Service",
        })

    return {"data": resultado}


@app.get("/climatologia")
def climatologia(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    """
    Climatologia mensal (12 meses) do ponto da grade mais próximo da coordenada.
    Retorna 404 se a coordenada estiver longe da área coberta.
    """
    exige_legado()
    dist = np.sqrt((_lats - latitude) ** 2 + (_lons - longitude) ** 2)
    i = int(dist.argmin())

    if dist[i] > DIST_MAX_GRAUS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Sem ponto de dados a menos de {DIST_MAX_GRAUS}° de ({latitude}, {longitude}). "
                f"Ponto mais próximo: {_lats[i]:.4f}, {_lons[i]:.4f}, a {dist[i]:.2f}°. "
                f"Área dos dados: lat {_lats.min():.2f} a {_lats.max():.2f}, "
                f"lon {_lons.min():.2f} a {_lons.max():.2f}."
            ),
        )

    lat_encontrada, lon_encontrada = float(_lats[i]), float(_lons[i])

    resultado_por_mes = {}
    for mes in range(1, 13):
        chave = (lat_encontrada, lon_encontrada, mes)
        if chave not in _clim.index:
            continue
        linha = _clim.loc[chave]
        resultado_por_mes[mes] = {
            "clorofila_media_historica_mes": num(linha["clorofila_media_historica_mes"], 4),
            "salinidade_media_historica_mes": num(linha["salinidade_media_historica_mes"], 3),
        }

    return {
        "latitude_encontrada": round(lat_encontrada, 4),
        "longitude_encontrada": round(lon_encontrada, 4),
        "climatologia_por_mes": resultado_por_mes,
    }


@app.get("/predictions")
def predictions(limit: int = Query(100, ge=1, le=5000), horizon: int = Query(1, ge=1, le=12)):
    """
    Previsão oficial (saidas/ULTIMA.json, a mesma do /forecast) para o mês t0+horizon.
    Padrão: o próximo mês depois do último observado. Pontos sem dado suficiente ficam de fora.
    """
    prev, meta, _ = dados_previsao()
    sel = prev[(prev["forecast_horizon"] == horizon) & prev["bloom_probability"].notna()]
    amostra = espalha(sel, limit)

    resultado = []
    for _, linha in amostra.iterrows():
        ano, mes = str(linha["target_date"]).split("-")[:2]
        resultado.append({
            "latitude": num(linha["latitude"], 4),
            "longitude": num(linha["longitude"], 4),
            "year": int(ano),
            "month": int(mes),
            "probability": num(linha["bloom_probability"], 4),
            "risk_level": linha["risk_level"],
            "model_version": linha["model_version"],
            "tipo": "previsao",
            "issue_month": meta["issue_month"],
            "forecast_horizon": int(linha["forecast_horizon"]),
            "region": linha["region"],
            "climatological_probability": num(linha["climatological_probability"], 4),
        })

    return {"predictions": resultado}


@app.post("/predict")
def prever(dados: DadosSazonais):
    """Classifica um conjunto de features digitado (modelo antigo, sem horizonte). Para o futuro use /forecast."""
    exige_legado()
    valores = dados.model_dump()

    if not 1 <= valores["mes"] <= 12:
        raise HTTPException(status_code=422, detail="'mes' deve estar entre 1 e 12.")

    entrada = pd.DataFrame([valores])[features]
    probabilidade = float(modelo.predict_proba(entrada)[0][1])

    return {
        "probability": round(probabilidade, 4),
        "risk": "ALTO" if probabilidade >= limiar else "BAIXO",
        "model_version": versao,
    }
