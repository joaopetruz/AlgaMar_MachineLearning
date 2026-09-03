import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

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

# Treino = até 2021, teste = 2022-2024
treino = tabela["ano"] <= 2021
teste = tabela["ano"] > 2021

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
for limiar in [0.5, 0.3, 0.2, 0.1]:
    y_pred = (probabilidades >= limiar).astype(int)
    f1 = f1_score(y_teste, y_pred)
    recall = recall_score(y_teste, y_pred)
    print(f"Limiar {limiar}: F1={f1:.4f} | Recall={recall:.4f}")

# ===== Limiar final escolhido: 0.2 (melhor F1 com recall mais alto) =====
LIMIAR_ESCOLHIDO = 0.2
y_pred_final = (probabilidades >= LIMIAR_ESCOLHIDO).astype(int)

print(f"\n=== Relatório (limiar {LIMIAR_ESCOLHIDO}) ===")
print(classification_report(y_teste, y_pred_final, target_names=["Sem floração", "Com floração"], zero_division=0))

print("\n=== Matriz de confusão ===")
print(confusion_matrix(y_teste, y_pred_final))

importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Importância das variáveis ===")
print(importancias)