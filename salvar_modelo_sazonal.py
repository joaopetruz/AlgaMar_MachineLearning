import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

tabela = pd.read_csv("dados/dados_features_sazonal_sp_completo.csv")

features = [
    "mes", "latitude", "longitude", "dist_hotspot",
    "clorofila_media_ano_anterior", "clorofila_maxima_ano_anterior",
    "temperatura_ano_anterior", "salinidade_ano_anterior", "vento_velocidade_ano_anterior",
    "clorofila_media_historica_mes", "salinidade_media_historica_mes", "vento_media_historica_mes",
    "clorofila_media_3anos", "salinidade_media_3anos", "vento_media_3anos",
]

X = tabela[features]
y = tabela["floracao"]

print("Treinando modelo final (sazonal, com salinidade e vento)...")
modelo = RandomForestClassifier(
    n_estimators=300, class_weight="balanced_subsample",
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
modelo.fit(X, y)
print("Treinamento concluído!")

joblib.dump(modelo, "modelo_floracao_sazonal.pkl")
joblib.dump(features, "features_modelo_sazonal.pkl")
joblib.dump(0.2, "limiar_decisao_sazonal.pkl")
joblib.dump("5.0.0-sazonal-25anos-completo", "versao_modelo.pkl")
print("Modelo, features, limiar e versão salvos com sucesso!")