import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

tabela = pd.read_csv("dados/dados_features_sazonal_sp.csv")

features = [
    "mes",
    "latitude",
    "longitude",
    "dist_hotspot",
    "clorofila_media_ano_anterior",
    "clorofila_maxima_ano_anterior",
    "temperatura_ano_anterior",
    "clorofila_media_historica_mes",
    "clorofila_media_3anos",
]

X = tabela[features]
y = tabela["floracao"]

print("Treinando modelo final (sazonal, todos os dados)...")
modelo = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
modelo.fit(X, y)
print("Treinamento concluído!")

joblib.dump(modelo, "modelo_floracao_sazonal.pkl")
print("Modelo salvo como: modelo_floracao_sazonal.pkl")

joblib.dump(features, "features_modelo_sazonal.pkl")
print("Features salvas como: features_modelo_sazonal.pkl")

LIMIAR_DECISAO = 0.2
joblib.dump(LIMIAR_DECISAO, "limiar_decisao_sazonal.pkl")
print(f"Limiar de decisão salvo: {LIMIAR_DECISAO}")

VERSAO_MODELO = "3.0.0-sazonal-25anos"
joblib.dump(VERSAO_MODELO, "versao_modelo.pkl")
print(f"Versão do modelo salva: {VERSAO_MODELO}")