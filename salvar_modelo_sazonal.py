import pandas as pd
import joblib
from xgboost import XGBClassifier

tabela = pd.read_csv("dados/dados_features_sazonal_sp_com_salinidade.csv")

features = [
    "mes", "latitude", "longitude", "dist_hotspot",
    "clorofila_media_ano_anterior", "clorofila_maxima_ano_anterior",
    "temperatura_ano_anterior", "salinidade_ano_anterior",
    "clorofila_media_historica_mes", "salinidade_media_historica_mes",
    "clorofila_media_3anos", "salinidade_media_3anos",
]

X = tabela[features]
y = tabela["floracao"]

peso = (y == 0).sum() / (y == 1).sum()

print("Treinando modelo final (XGBoost, salinidade)...")
modelo = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=peso, random_state=42, n_jobs=-1,
    eval_metric="logloss"
)
modelo.fit(X, y)
print("Treinamento concluído!")

joblib.dump(modelo, "modelo_floracao_sazonal.pkl")
joblib.dump(features, "features_modelo_sazonal.pkl")
joblib.dump(0.5, "limiar_decisao_sazonal.pkl")
joblib.dump("7.0.0-xgboost-sazonal-salinidade", "versao_modelo.pkl")
print("Modelo, features, limiar e versão salvos com sucesso!")