import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Usar o arquivo completo (com todas as variáveis: salinidade, vento, correntes, nutrientes)
tabela = pd.read_csv("dados/dados_features_sazonal_sp_completo.csv")

# Todas as variáveis candidatas (excluindo identificadores e o target)
colunas_excluir = ["latitude", "longitude", "ano", "clorofila_media", "clorofila_maxima",
                    "temperatura_media", "salinidade_media", "vento_velocidade",
                    "corrente_velocidade", "nitrato", "fosfato", "oxigenio", "ph", "floracao"]

features_candidatas = [c for c in tabela.columns if c not in colunas_excluir]
print(f"Total de variáveis candidatas: {len(features_candidatas)}")
print(features_candidatas)

X = tabela[features_candidatas]
y = tabela["floracao"]

# Usar só uma amostra para acelerar o RFE (ele treina MUITOS modelos internamente)
amostra = tabela.sample(n=100000, random_state=42)
X_amostra = amostra[features_candidatas]
y_amostra = amostra["floracao"]

print("\nIniciando RFE com validação cruzada (isso vai demorar bastante)...")

modelo_base = RandomForestClassifier(
    n_estimators=100,  # menor para acelerar o processo de seleção
    class_weight="balanced_subsample",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

divisao = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

seletor = RFECV(
    estimator=modelo_base,
    step=1,              # remove 1 variável por vez
    cv=divisao,
    scoring="roc_auc",
    min_features_to_select=5,
    n_jobs=-1,
    verbose=2
)

seletor.fit(X_amostra, y_amostra)

print(f"\nNúmero ótimo de variáveis: {seletor.n_features_}")

variaveis_selecionadas = [f for f, selecionada in zip(features_candidatas, seletor.support_) if selecionada]
print(f"\nVariáveis selecionadas pelo RFE:")
for v in variaveis_selecionadas:
    print(f"  - {v}")

variaveis_descartadas = [f for f, selecionada in zip(features_candidatas, seletor.support_) if not selecionada]
print(f"\nVariáveis descartadas pelo RFE:")
for v in variaveis_descartadas:
    print(f"  - {v}")

import joblib
joblib.dump(variaveis_selecionadas, "features_selecionadas_rfe.pkl")
print("\nLista de features selecionadas salva em: features_selecionadas_rfe.pkl")