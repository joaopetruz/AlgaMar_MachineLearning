import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score, classification_report

# ===== Cenário 1: Só salinidade =====
print("="*60)
print("CENÁRIO 1: XGBoost com salinidade (clorofila+temp+sal)")
print("="*60)

tabela1 = pd.read_csv("dados/dados_features_sazonal_sp_com_salinidade.csv")
features1 = [
    "mes", "latitude", "longitude", "dist_hotspot",
    "clorofila_media_ano_anterior", "clorofila_maxima_ano_anterior",
    "temperatura_ano_anterior", "salinidade_ano_anterior",
    "clorofila_media_historica_mes", "salinidade_media_historica_mes",
    "clorofila_media_3anos", "salinidade_media_3anos",
]

X1 = tabela1[features1]
y1 = tabela1["floracao"]
treino1 = tabela1["ano"] <= 2021
teste1 = tabela1["ano"] > 2021

# Calcular peso de balanceamento (XGBoost não tem class_weight='balanced' nativo)
peso1 = (y1[treino1] == 0).sum() / (y1[treino1] == 1).sum()

modelo1 = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=peso1, random_state=42, n_jobs=-1,
    eval_metric="logloss"
)
modelo1.fit(X1[treino1], y1[treino1])

probs1 = modelo1.predict_proba(X1[teste1])[:, 1]
print("\nTestando limiares:")
for limiar in [0.5, 0.3, 0.2, 0.1]:
    pred = (probs1 >= limiar).astype(int)
    print(f"Limiar {limiar}: F1={f1_score(y1[teste1], pred):.4f} | Recall={recall_score(y1[teste1], pred):.4f}")

# ===== Cenário 2: Todas as variáveis (completo) =====
print("\n" + "="*60)
print("CENÁRIO 2: XGBoost com TODAS as variáveis")
print("="*60)

tabela2 = pd.read_csv("dados/dados_features_sazonal_sp_completo.csv")
colunas_excluir = ["latitude", "longitude", "ano", "clorofila_media", "clorofila_maxima",
                    "temperatura_media", "salinidade_media", "vento_velocidade",
                    "corrente_velocidade", "nitrato", "fosfato", "oxigenio", "ph", "floracao"]
features2 = [c for c in tabela2.columns if c not in colunas_excluir]

X2 = tabela2[features2]
y2 = tabela2["floracao"]
treino2 = tabela2["ano"] <= 2021
teste2 = tabela2["ano"] > 2021

peso2 = (y2[treino2] == 0).sum() / (y2[treino2] == 1).sum()

modelo2 = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=peso2, random_state=42, n_jobs=-1,
    eval_metric="logloss"
)
modelo2.fit(X2[treino2], y2[treino2])

probs2 = modelo2.predict_proba(X2[teste2])[:, 1]
print("\nTestando limiares:")
for limiar in [0.5, 0.3, 0.2, 0.1]:
    pred = (probs2 >= limiar).astype(int)
    print(f"Limiar {limiar}: F1={f1_score(y2[teste2], pred):.4f} | Recall={recall_score(y2[teste2], pred):.4f}")

# ===== Cenário 3: Features selecionadas pelo RFE =====
print("\n" + "="*60)
print("CENÁRIO 3: XGBoost com as 25 features do RFE")
print("="*60)

features3 = joblib.load("features_selecionadas_rfe.pkl")
X3 = tabela2[features3]
y3 = tabela2["floracao"]

modelo3 = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=peso2, random_state=42, n_jobs=-1,
    eval_metric="logloss"
)
modelo3.fit(X3[treino2], y3[treino2])

probs3 = modelo3.predict_proba(X3[teste2])[:, 1]
print("\nTestando limiares:")
for limiar in [0.5, 0.3, 0.2, 0.1]:
    pred = (probs3 >= limiar).astype(int)
    print(f"Limiar {limiar}: F1={f1_score(y3[teste2], pred):.4f} | Recall={recall_score(y3[teste2], pred):.4f}")

print("\n" + "="*60)
print("RESUMO COMPARATIVO (Random Forest já testado antes):")
print("RF  - Só salinidade:  F1=0.671 | Recall=82.5%")
print("RF  - Todas var.:     F1=0.615 | Recall=72.0%")
print("RF  - RFE (25 vars):  F1=0.623 | Recall=73.1%")
print("="*60)
