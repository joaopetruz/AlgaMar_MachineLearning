import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Carregar dados
tabela = pd.read_csv("dados/dados_features_2024.csv", parse_dates=["time"])

features = [
    "temperatura_celsius",
    "mes",
    "dia_do_ano",
    "clorofila_lag1",
    "clorofila_lag3",
    "clorofila_lag7",
    "temperatura_lag1",
    "clorofila_media_7d",
    "clorofila_media_14d",
    "temperatura_media_7d",
    "dist_hotspot",
    "latitude",
    "longitude",
]

X = tabela[features]
y = tabela["floracao"]

# Treinar o modelo final usando TODOS os dados disponíveis
# (agora que já validamos com o split treino/teste, usamos tudo para o modelo "de produção")
print("Treinando modelo final com todos os dados...")
modelo = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
modelo.fit(X, y)
print("Treinamento concluído!")

# Salvar o modelo em disco
joblib.dump(modelo, "modelo_floracao.pkl")
print("\nModelo salvo como: modelo_floracao.pkl")

# Salvar também a lista de features (importante para saber a ordem/nomes depois)
joblib.dump(features, "features_modelo.pkl")
print("Lista de features salva como: features_modelo.pkl")

# Salvar o limiar de decisão escolhido, para reutilizar depois
LIMIAR_DECISAO = 0.2
joblib.dump(LIMIAR_DECISAO, "limiar_decisao.pkl")
print(f"Limiar de decisão salvo: {LIMIAR_DECISAO}")