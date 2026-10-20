"""Monta o painel mensal (histórico da equipe 2000–2024 + meses baixados das APIs).

Antes: python atualizar_dados_atuais.py   (baixa os meses novos da Copernicus)
"""
import json

from algamar_ml import config
from algamar_ml.processamento.painel import constroi, salva

cfg = config.carrega()
painel, manifesto = constroi(cfg)
salva(painel, manifesto, cfg["caminhos"]["painel"])
resumo = {k: v for k, v in manifesto.items() if k != "meses_novos"}
resumo["meses_novos"] = sorted({f"{e['ano']}-{e['mes']:02d}" for e in manifesto["meses_novos"]})
print(json.dumps(resumo, ensure_ascii=False, indent=1))
print(painel.groupby("ano").size().tail(4))
print(painel[painel.ano >= 2025].groupby(["ano", "mes"])[["fonte_clorofila", "fonte_temperatura", "fonte_salinidade"]].first())
