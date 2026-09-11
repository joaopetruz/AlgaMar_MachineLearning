from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API de Previsão de Floração de Algas - AlgarMar (Litoral SP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo e configurações salvas
modelo = joblib.load("modelo_floracao_sazonal.pkl")
features = joblib.load("features_modelo_sazonal.pkl")
limiar = joblib.load("limiar_decisao_sazonal.pkl")
versao = joblib.load("versao_modelo.pkl")

# Carregar a base de dados ambientais disponível
dados_ambientais = pd.read_csv("dados/dados_features_sazonal_sp.csv")

class DadosSazonais(BaseModel):
    mes: int
    latitude: float
    longitude: float
    dist_hotspot: float
    clorofila_media_ano_anterior: float
    clorofila_maxima_ano_anterior: float
    temperatura_ano_anterior: float
    salinidade_ano_anterior: float
    clorofila_media_historica_mes: float
    salinidade_media_historica_mes: float
    clorofila_media_3anos: float
    salinidade_media_3anos: float


@app.get("/")
def raiz():
    return {"mensagem": "API AlgarMar funcionando! Acesse /docs para testar."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/marine-data")
def marine_data(limit: int = 100):
    amostra = dados_ambientais.tail(limit)

    resultado = []
    for _, linha in amostra.iterrows():
        # Verifica com segurança se as colunas existem no DataFrame atual
        salinidade = linha.get("salinidade_media", linha.get("salinidade", 0.0))
        temp = linha.get("temperatura_media", linha.get("temperatura", 0.0))
        cloro_med = linha.get("clorofila_media", 0.0)
        cloro_max = linha.get("clorofila_maxima", 0.0)

        resultado.append({
            "latitude": round(float(linha["latitude"]), 4),
            "longitude": round(float(linha["longitude"]), 4),
            "year": int(linha["ano"]),
            "month": int(linha["mes"]),
            "temperature_celsius": round(float(temp), 2),
            "chlorophyll_mg_m3": round(float(cloro_med), 4),
            "chlorophyll_max_mg_m3": round(float(cloro_max), 4),
            "salinity_psu": round(float(salinidade), 3),
            "source": "Copernicus Marine Service"
        })

    return {"data": resultado}


@app.get("/predictions")
def predictions(limit: int = 100):
    amostra = dados_ambientais.tail(limit).copy()

    X = amostra[features]
    probabilidades = modelo.predict_proba(X)[:, 1]

    resultado = []
    for i, (_, linha) in enumerate(amostra.iterrows()):
        prob = float(probabilidades[i])
        risco = "ALTO" if prob >= limiar else "BAIXO"

        resultado.append({
            "latitude": round(float(linha["latitude"]), 4),
            "longitude": round(float(linha["longitude"]), 4),
            "year": int(linha["ano"]),
            "month": int(linha["mes"]),
            "probability": round(prob, 4),
            "risk_level": risco,
            "model_version": versao
        })

    return {"predictions": resultado}


@app.post("/predict")
def prever(dados: DadosSazonais):
    entrada = pd.DataFrame([dados.model_dump()])[features]
    probabilidade = modelo.predict_proba(entrada)[0][1]
    risco = "ALTO" if probabilidade >= limiar else "BAIXO"

    return {
        "probability": round(float(probabilidade), 4),
        "risk": risco,
        "model_version": versao
    }