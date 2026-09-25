import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

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

treino = tabela["ano"] <= 2021
teste = tabela["ano"] > 2021

X_treino, X_teste = X[treino], X[teste]
y_treino, y_teste = y[treino], y[teste]

peso = (y_treino == 0).sum() / (y_treino == 1).sum()

print(f"Linhas de treino: {len(X_treino)}")
print(f"Linhas de teste: {len(X_teste)}")

print("\nTreinando XGBoost...")
modelo = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=peso, random_state=42, n_jobs=-1,
    eval_metric="logloss"
)
modelo.fit(X_treino, y_treino)
print("Treinamento concluído!")

probabilidades = modelo.predict_proba(X_teste)[:, 1]

print("\n=== Testando diferentes limiares de decisão ===")
for limiar in [0.6, 0.5, 0.4, 0.3, 0.2]:
    y_pred = (probabilidades >= limiar).astype(int)
    print(f"Limiar {limiar}: F1={f1_score(y_teste, y_pred):.4f} | Recall={recall_score(y_teste, y_pred):.4f}")

LIMIAR_ESCOLHIDO = 0.5
y_pred_final = (probabilidades >= LIMIAR_ESCOLHIDO).astype(int)

print(f"\n=== Relatório (limiar {LIMIAR_ESCOLHIDO}) ===")
print(classification_report(y_teste, y_pred_final, target_names=["Sem floração", "Com floração"], zero_division=0))

print("\n=== Matriz de confusão ===")
print(confusion_matrix(y_teste, y_pred_final))

importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Importância das variáveis ===")
print(importancias)