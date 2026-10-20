import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

# Carregar dados tratados (antes do target, para recriar com limiares diferentes)
base = pd.read_csv("dados/dados_tratados_2024_sp.csv", parse_dates=["time"])
base = base.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)

base["mes"] = base["time"].dt.month
base["dia_do_ano"] = base["time"].dt.dayofyear

grupo = base.groupby(["latitude", "longitude"])
base["clorofila_lag1"] = grupo["clorofila"].shift(1)
base["clorofila_lag3"] = grupo["clorofila"].shift(3)
base["clorofila_lag7"] = grupo["clorofila"].shift(7)
base["temperatura_lag1"] = grupo["temperatura_celsius"].shift(1)
base["clorofila_media_7d"] = grupo["clorofila"].transform(lambda x: x.rolling(7, min_periods=1).mean())
base["clorofila_media_14d"] = grupo["clorofila"].transform(lambda x: x.rolling(14, min_periods=1).mean())
base["temperatura_media_7d"] = grupo["temperatura_celsius"].transform(lambda x: x.rolling(7, min_periods=1).mean())

hotspots = [(-23.98, -46.35), (-23.45, -45.75)]
distancias = []
for lat_h, lon_h in hotspots:
    dist = np.sqrt((base["latitude"] - lat_h) ** 2 + (base["longitude"] - lon_h) ** 2)
    distancias.append(dist)
base["dist_hotspot"] = pd.concat(distancias, axis=1).min(axis=1)

base = base.dropna(subset=["clorofila_lag1", "clorofila_lag3", "clorofila_lag7", "temperatura_lag1"])

features = [
    "temperatura_celsius", "mes", "dia_do_ano",
    "clorofila_lag1", "clorofila_lag3", "clorofila_lag7", "temperatura_lag1",
    "clorofila_media_7d", "clorofila_media_14d", "temperatura_media_7d",
    "dist_hotspot", "latitude", "longitude",
]

data_corte = "2024-11-01"
treino_mask = base["time"] < data_corte
teste_mask = base["time"] >= data_corte

print("=== Testando diferentes limiares de definição de floração ===\n")

for limite in [1.5, 2.0, 3.0]:
    y = (base["clorofila"] > limite).astype(int)

    X_treino, X_teste = base.loc[treino_mask, features], base.loc[teste_mask, features]
    y_treino, y_teste = y[treino_mask], y[teste_mask]

    modelo = RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    modelo.fit(X_treino, y_treino)
    probs = modelo.predict_proba(X_teste)[:, 1]

    print(f"--- LIMITE_FLORACAO = {limite} (casos positivos: {y.sum()}) ---")
    for limiar_decisao in [0.2, 0.1, 0.05]:
        pred = (probs >= limiar_decisao).astype(int)
        f1 = f1_score(y_teste, pred)
        recall = recall_score(y_teste, pred)
        print(f"  Decisão {limiar_decisao}: F1={f1:.4f} | Recall={recall:.4f}")
    print()