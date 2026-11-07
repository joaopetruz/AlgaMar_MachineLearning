import json

import joblib
import pandas as pd

from algamar_ml.avaliacao import novo_modelo

# O limiar vem da validação (treinar_modelo_sazonal.py), não do teste
with open("resultado_avaliacao_sazonal.json", encoding="utf-8") as f:
    avaliacao = json.load(f)
features = avaliacao["features"]
limiar = avaliacao["limiar_escolhido_na_validacao"]

tabela = pd.read_csv("dados/dados_features_sazonal_sp_com_salinidade.csv")
X = tabela[features]
y = tabela["floracao"]

print(f"Treinando modelo final (XGBoost, {len(features)} features, todos os anos, limiar {limiar})...")
modelo = novo_modelo(y)
modelo.fit(X, y)
print("Treinamento concluído!")

joblib.dump(modelo, "modelo_floracao_sazonal.pkl")
joblib.dump(features, "features_modelo_sazonal.pkl")
joblib.dump(limiar, "limiar_decisao_sazonal.pkl")
joblib.dump("8.0.0-xgboost-sazonal-sem-vazamento", "versao_modelo.pkl")
print("Modelo, features, limiar e versão salvos com sucesso!")
