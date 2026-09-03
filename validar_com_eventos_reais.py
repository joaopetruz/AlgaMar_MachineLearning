import pandas as pd
import joblib

base = pd.read_csv("dados/dados_features_sazonal_sp_com_eventos_reais.csv")

modelo = joblib.load("modelo_floracao_sazonal.pkl")
features = joblib.load("features_modelo_sazonal.pkl")
limiar = joblib.load("limiar_decisao_sazonal.pkl")

# Pegar só os pontos com floração real confirmada
eventos_reais = base[base["floracao_confirmada_real"] == 1].copy()

print(f"Validando o modelo contra {len(eventos_reais)} eventos reais confirmados:\n")

X_eventos = eventos_reais[features]
probabilidades = modelo.predict_proba(X_eventos)[:, 1]
eventos_reais["probabilidade_prevista"] = probabilidades
eventos_reais["classificado_como_risco"] = probabilidades >= limiar

colunas_exibir = ["ano", "mes", "latitude", "longitude", "probabilidade_prevista", "classificado_como_risco"]
print(eventos_reais[colunas_exibir].to_string(index=False))

acertos = eventos_reais["classificado_como_risco"].sum()
print(f"\nO modelo classificou corretamente como 'risco' em {acertos} de {len(eventos_reais)} eventos reais confirmados.")
print(f"Taxa de acerto nos eventos reais: {acertos/len(eventos_reais)*100:.1f}%")