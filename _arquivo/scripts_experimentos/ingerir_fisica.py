"""Leva a saída do solver físico para a feature store do ML (camada intermediária).

Uso:
  python ingerir_fisica.py tabela <arquivo.parquet|csv> [--fonte croco_d2] [--versao ...]   # export do cfd_navier_stokes (contrato 13.0)
  python ingerir_fisica.py grade  <arquivo.nc> --u u --v v [--t temp] [--fonte navier_stokes]  # campo u,v em grade lat/lon
"""
import argparse
import json

import pandas as pd

from algamar_ml import config
from algamar_ml.fisica import feature_store
from algamar_ml.fisica.fontes import FonteGradeUV, FonteTabelaContrato13

ap = argparse.ArgumentParser()
ap.add_argument("tipo", choices=["tabela", "grade"])
ap.add_argument("arquivo")
ap.add_argument("--fonte", default=None)
ap.add_argument("--versao", default="desconhecida")
ap.add_argument("--u", default="u")
ap.add_argument("--v", default="v")
ap.add_argument("--t", default=None)
ap.add_argument("--dist-max-km", type=float, default=3.0)
args = ap.parse_args()

cfg = config.carrega()
pontos = pd.read_parquet(cfg["caminhos"]["painel"], columns=["latitude", "longitude"]).drop_duplicates()
if args.tipo == "tabela":
    fonte = FonteTabelaContrato13(args.arquivo, nome=args.fonte or "croco_d2", versao=args.versao)
else:
    fonte = FonteGradeUV(args.arquivo, nome=args.fonte or "navier_stokes", versao=args.versao,
                         var_u=args.u, var_v=args.v, var_t=args.t)
_, manifesto = feature_store.ingere(fonte, pontos, cfg["caminhos"]["feature_store"], dist_max_km=args.dist_max_km)
print(json.dumps(manifesto, ensure_ascii=False, indent=1))
