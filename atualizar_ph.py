"""Baixa o pH de superfície (modelo Copernicus) dos últimos meses e grava a tabela por município do dashboard.

  python atualizar_ph.py              # os 12 meses do painel + o mês seguinte (o "mês atual" do dashboard)
  python atualizar_ph.py --meses 24   # mais meses para trás

Usa o login da Copernicus já configurado (o mesmo do atualizar_dados_atuais.py). Detalhes: algamar_ml/ph.py.
"""
import argparse

import pandas as pd

from algamar_ml import config, ph
from algamar_ml.api_dashboard import MUNICIPIOS

ap = argparse.ArgumentParser()
ap.add_argument("--meses", type=int, default=12)
a = ap.parse_args()

cfg = config.carrega()
painel = pd.read_parquet(cfg["caminhos"]["painel"], columns=["ano", "mes"])
ultimo = int((painel["ano"] * 12 + painel["mes"] - 1).max())
meses = [(t // 12, t % 12 + 1) for t in range(ultimo - a.meses + 1, ultimo + 2)]   # + o mês seguinte
print(f"baixando pH de {meses[0][0]}-{meses[0][1]:02d} a {meses[-1][0]}-{meses[-1][1]:02d} ({len(meses)} meses)")
arquivos = ph.baixa(cfg, meses)
tabela = ph.por_municipio(arquivos, MUNICIPIOS)
destino = ph.caminho_tabela(cfg)
tabela.to_parquet(destino, index=False)
print(f"{len(tabela)} linhas → {destino}")
print(tabela.groupby("municipio")[["ph", "dist_km"]].mean().round(3).to_string())
