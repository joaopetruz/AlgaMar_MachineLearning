import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import create_model

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


def carregar(nome: str):
    caminho = BASE_DIR / nome
    if not caminho.exists():
        raise RuntimeError(
            f"Arquivo '{nome}' não encontrado em {BASE_DIR}. "
            "Rode salvar_modelo_sazonal.py antes de iniciar a API."
        )
    return joblib.load(caminho)


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

# Último mês disponível na base
_ano_max = int(dados_ambientais["ano"].max())
_mes_max = int(dados_ambientais.loc[dados_ambientais["ano"] == _ano_max, "mes"].max())
_ultimo_mes = dados_ambientais[
    (dados_ambientais["ano"] == _ano_max) & (dados_ambientais["mes"] == _mes_max)
].reset_index(drop=True)


def amostra_recente(limit: int) -> pd.DataFrame:
    """
    Pontos do mês mais recente. Se limit for menor que o total, escolhe pontos
    espaçados ao longo de toda a grade (ordenada por latitude/longitude), para
    cobrir a costa inteira e não só o extremo norte.
    """
    if limit >= len(_ultimo_mes):
        return _ultimo_mes
    posicoes = np.linspace(0, len(_ultimo_mes) - 1, limit).round().astype(int)
    return _ultimo_mes.iloc[posicoes]


# Grade de pontos únicos e climatologia por ponto/mês, calculadas uma vez
_pontos = dados_ambientais[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
_lats = _pontos["latitude"].to_numpy()
_lons = _pontos["longitude"].to_numpy()

_clim = (
    dados_ambientais
    .groupby(["latitude", "longitude", "mes"])[
        ["clorofila_media_historica_mes", "salinidade_media_historica_mes"]
    ]
    .first()
)

# Schema do /predict gerado a partir das features do modelo salvo
# (se a lista de features mudar, a API acompanha sozinha)
DadosSazonais = create_model(
    "DadosSazonais",
    **{f: ((int if f == "mes" else float), ...) for f in features},
)


def num(valor, casas):
    """Converte para float arredondado; NaN vira None (NaN quebra o JSON)."""
    valor = float(valor)
    return None if math.isnan(valor) else round(valor, casas)


@app.get("/")
def raiz():
    return {"mensagem": "API AlgarMar (XGBoost sazonal, salinidade) funcionando! Acesse /docs para testar."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/marine-data")
def marine_data(limit: int = Query(100, ge=1, le=5000)):
    """Dados ambientais do mês mais recente, distribuídos por toda a costa."""
    amostra = amostra_recente(limit)

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
def predictions(limit: int = Query(100, ge=1, le=5000)):
    """Previsões de risco do mês mais recente, distribuídas por toda a costa."""
    amostra = amostra_recente(limit)

    probabilidades = modelo.predict_proba(amostra[features])[:, 1]

    resultado = []
    for prob, (_, linha) in zip(probabilidades, amostra.iterrows()):
        prob = float(prob)
        resultado.append({
            "latitude": num(linha["latitude"], 4),
            "longitude": num(linha["longitude"], 4),
            "year": int(linha["ano"]),
            "month": int(linha["mes"]),
            "probability": round(prob, 4),
            "risk_level": "ALTO" if prob >= limiar else "BAIXO",
            "model_version": versao,
        })

    return {"predictions": resultado}


@app.post("/predict")
def prever(dados: DadosSazonais):
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
