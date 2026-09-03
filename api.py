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

# Carregar a base de dados ambientais processada (uma vez, ao iniciar a API)
dados_ambientais = pd.read_csv("dados/dados_features_sazonal_sp.csv")

class DadosSazonais(BaseModel):
    mes: int
    latitude: float
    longitude: float
    dist_hotspot: float
    clorofila_media_ano_anterior: float
    clorofila_maxima_ano_anterior: float
    temperatura_ano_anterior: float
    clorofila_media_historica_mes: float
    clorofila_media_3anos: float


@app.get("/")
def raiz():
    return {"mensagem": "API AlgarMar (modelo sazonal) funcionando! Acesse /docs para testar."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/marine-data")
def marine_data(limit: int = 100):
    """
    Retorna os dados ambientais coletados (agregados mensalmente).
    Parâmetro opcional 'limit' controla quantos registros retornar (padrão: 100).
    """
    amostra = dados_ambientais.tail(limit)

    resultado = []
    for _, linha in amostra.iterrows():
        resultado.append({
            "latitude": round(float(linha["latitude"]), 4),
            "longitude": round(float(linha["longitude"]), 4),
            "year": int(linha["ano"]),
            "month": int(linha["mes"]),
            "temperature_celsius": round(float(linha["temperatura_media"]), 2),
            "chlorophyll_mg_m3": round(float(linha["clorofila_media"]), 4),
            "chlorophyll_max_mg_m3": round(float(linha["clorofila_maxima"]), 4),
            "source": "Copernicus Marine Service"
        })

    return {"data": resultado}


@app.get("/predictions")
def predictions(limit: int = 100):
    """
    Retorna previsões de risco de floração já calculadas para os dados mais recentes.
    Parâmetro opcional 'limit' controla quantos registros retornar (padrão: 100).
    """
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