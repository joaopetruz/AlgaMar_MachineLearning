import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import numpy as np

tabela = pd.read_csv("dados/dados_features_2024_sp.csv", parse_dates=["time"])

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

data_corte = "2024-11-01"
treino = tabela["time"] < data_corte
teste = tabela["time"] >= data_corte

X_treino, X_teste = X[treino], X[teste]
y_treino, y_teste = y[treino], y[teste]

print(f"Linhas de treino: {len(X_treino)}")
print(f"Linhas de teste: {len(X_teste)}")
print(f"Casos de floração no treino: {y_treino.sum()}")
print(f"Casos de floração no teste: {y_teste.sum()}")

print("\nTreinando Random Forest...")
modelo = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_treino, y_treino)
print("Treinamento concluído!")

probabilidades = modelo.predict_proba(X_teste)[:, 1]

print("\n=== Testando diferentes limiares de decisão ===")
for limiar in [0.5, 0.3, 0.2, 0.1, 0.05]:
    y_pred = (probabilidades >= limiar).astype(int)
    f1 = f1_score(y_teste, y_pred)
    recall = recall_score(y_teste, y_pred)
    print(f"Limiar {limiar}: F1={f1:.4f} | Recall={recall:.4f}")

LIMIAR_ESCOLHIDO = 0.2
y_pred_final = (probabilidades >= LIMIAR_ESCOLHIDO).astype(int)

print(f"\n=== Relatório de classificação (limiar {LIMIAR_ESCOLHIDO}) ===")
print(classification_report(y_teste, y_pred_final, target_names=["Sem floração", "Com floração"], zero_division=0))

print("\n=== Matriz de confusão ===")
print(confusion_matrix(y_teste, y_pred_final))

importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Importância das variáveis ===")
print(importancias)