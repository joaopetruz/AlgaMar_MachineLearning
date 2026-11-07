import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

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

treino = tabela["ano"] <= 2021
X_treino, y_treino = X[treino], y[treino]

grade_parametros = {
    "n_estimators": [200, 300],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [2, 5],
}

# Embaralhar os dados antes de dividir em folds (corrige o problema de localizações "novas")
divisao = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Iniciando busca de melhores hiperparâmetros (isso pode demorar vários minutos)...")

busca = GridSearchCV(
    estimator=RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    param_grid=grade_parametros,
    scoring="roc_auc",
    cv=divisao,
    verbose=2,
    n_jobs=-1
)

busca.fit(X_treino, y_treino)

print("\n=== Melhor combinação encontrada ===")
print(busca.best_params_)
print(f"\nMelhor ROC-AUC (validação cruzada): {busca.best_score_:.4f}")

joblib.dump(busca.best_params_, "melhores_hiperparametros.pkl")
print("\nMelhores hiperparâmetros salvos em: melhores_hiperparametros.pkl")