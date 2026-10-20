import argparse
import json

import pandas as pd

from algamar_ml.croco import junta_croco, le_croco_mensal

ap = argparse.ArgumentParser(description="Junta as features mensais do CROCO (Navier-Stokes) ao dataset do ML.")
ap.add_argument("croco", help="arquivo mensal do CROCO (Parquet do contrato 13.0), ex.: dados/croco/croco_features_mensal.parquet")
ap.add_argument("--conjunto", default="CROCO_CORE", help="conjunto em contrato_croco/feature_sets.json")
ap.add_argument("--modo", default="FORECAST_SAZONAL", help="NOWCAST, FORECAST_SAZONAL ou FORECAST_1M")
ap.add_argument("--entrada", default="dados/dados_features_sazonal_sp_com_salinidade.csv")
ap.add_argument("--saida", default="dados/dados_features_sazonal_sp_croco.csv")
args = ap.parse_args()

ml = pd.read_csv(args.entrada)
croco = le_croco_mensal(args.croco)
tabela, relatorio = junta_croco(ml, croco, args.conjunto, args.modo)

print(json.dumps(relatorio, ensure_ascii=False, indent=1))
tabela.to_csv(args.saida, index=False)
with open(args.saida.replace(".csv", "_join_report.json"), "w", encoding="utf-8") as f:
    json.dump(relatorio, f, ensure_ascii=False, indent=1)
print(f"\nArquivo salvo: {args.saida}")
print(f"Para treinar: python treinar_modelo_sazonal.py --entrada {args.saida} --croco {args.conjunto} --modo {args.modo}")
