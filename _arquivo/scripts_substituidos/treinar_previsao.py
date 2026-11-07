"""Treina e avalia o modelo de previsão multi-horizonte (t0 → t0+1..t0+12).

Uso:
  python treinar_previsao.py --versao baseline-hist-api-v1
  python treinar_previsao.py --versao fisica-v1 --com-fisica          (quando a feature store existir)
As flags de config/ml.toml valem por padrão; as opções abaixo as sobrescrevem só nesta execução.
"""
import argparse
import json
from pathlib import Path

from algamar_ml import config, pipeline
from algamar_ml.modelos import treino as tr

ap = argparse.ArgumentParser()
ap.add_argument("--versao", required=True)
ap.add_argument("--com-fisica", action="store_true")
ap.add_argument("--sem-api", action="store_true")
ap.add_argument("--sem-historico", action="store_true")
args = ap.parse_args()

cfg = config.carrega()
if args.com_fisica:
    cfg["features"]["use_navier_stokes_features"] = True
if args.sem_api:
    cfg["features"]["use_api_features"] = False
if args.sem_historico:
    cfg["features"]["use_historical_features"] = False

cubo, _ = pipeline.carrega_cubo(cfg)
matrizes, nomes_fisica = pipeline.fisica(cfg, cubo)
import numpy as np  # noqa: E402
from algamar_ml import croco_ml  # noqa: E402
excluir = np.where(getattr(cubo, "croco_validade", np.array([])) == croco_ml.NAO_USAR)[0] if croco_ml.grupos_ativos(cfg["features"]) else None
linhas = pipeline.dataset(cfg, cubo, matrizes, excluir_nao_usar=excluir)
features = pipeline.features_de(cfg, nomes_fisica)
print(f"Linhas com alvo observado: {len(linhas):,} | features: {len(features)}")

manifesto_painel = json.loads(Path(cfg["caminhos"]["painel"]).with_suffix(".manifesto.json").read_text(encoding="utf-8"))
modelo, cartao, _ = pipeline.treina_e_avalia(
    cfg, linhas, features, args.versao, pipeline.flags_de(cfg),
    {"painel": {"sha256": manifesto_painel["sha256_painel"], "ultimo_mes": manifesto_painel["ultimo_mes"]}})
tr.salva(modelo, cartao, Path(cfg["caminhos"]["modelos"]) / args.versao)

g = cartao["teste"]["geral"]
print(f"\nLimiar (validação): {cartao['limiar_alto']}  | linhas {cartao['linhas']}")
print(f"TESTE {cartao['periodos']['teste']}: modelo PR-AUC {g['modelo']['pr_auc']:.3f} ROC {g['modelo']['roc_auc']:.3f} "
      f"Brier {g['modelo']['brier']:.3f} F1 {g['modelo']['f1']:.3f} | climatologia PR-AUC {g['climatologia']['pr_auc']:.3f} "
      f"| persistência PR-AUC {g['persistencia']['pr_auc']:.3f}")
print("\nh   PR-AUC modelo  climatologia  persistência   F1")
for h in map(str, range(1, 13)):
    m = cartao["teste"]["por_h"][h]
    print(f"{h:>2}   {m['modelo']['pr_auc']:.3f}         {m['climatologia']['pr_auc']:.3f}         {m['persistencia']['pr_auc']:.3f}        {m['modelo']['f1']:.3f}")
print(f"\nModelo salvo em {Path(cfg['caminhos']['modelos']) / args.versao}")
