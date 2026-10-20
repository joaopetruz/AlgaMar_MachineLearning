import argparse
import json

import pandas as pd

from algamar_ml.avaliacao import treina_e_avalia
from algamar_ml.croco import features_do_conjunto
from algamar_ml.features import FEATURES_BASELINE

ap = argparse.ArgumentParser(description="Treina no treino, escolhe o limiar na validação e avalia no teste.")
ap.add_argument("--entrada", default="dados/dados_features_sazonal_sp_com_salinidade.csv")
ap.add_argument("--croco", default=None, help="conjunto de features do CROCO a somar ao baseline (ex.: CROCO_TS)")
ap.add_argument("--modo", default="FORECAST_SAZONAL", help="NOWCAST, FORECAST_SAZONAL ou FORECAST_1M")
args = ap.parse_args()

tabela = pd.read_csv(args.entrada)
features = FEATURES_BASELINE + (features_do_conjunto(args.croco, args.modo) if args.croco else [])

print(f"Features ({len(features)}): {features}")
modelo, resultado, _ = treina_e_avalia(tabela, features)

for nome, n in resultado["linhas"].items():
    print(f"Linhas de {nome}: {n}")
print(f"\nLimiar escolhido na VALIDAÇÃO (máximo F1): {resultado['limiar_escolhido_na_validacao']}")
for parte in ("validacao", "teste"):
    m = resultado[parte]
    print(f"\n=== {parte} ===")
    print(f"PR-AUC={m['pr_auc']:.4f} | ROC-AUC={m['roc_auc']:.4f} | F1={m['f1']:.4f} | "
          f"Recall={m['recall']:.4f} | Precision={m['precision']:.4f}")
    print(f"Matriz de confusão: {m['matriz_confusao']}")

importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Importância das variáveis ===")
print(importancias)

resultado.update({"features": features, "croco": args.croco, "modo": args.modo})
with open("resultado_avaliacao_sazonal.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=1)
print("\nSalvo: resultado_avaliacao_sazonal.json (o salvar_modelo_sazonal.py lê o limiar daqui)")
