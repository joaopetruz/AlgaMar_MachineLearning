"""Gera a previsão dos próximos 12 meses a partir do último mês observado do painel.

Uso: python prever_12_meses.py --versao baseline-hist-api-v1
Saídas em previsoes/<mês de emissão>/: previsao_pontos (parquet+csv), resumo_regioes.csv,
pico_por_ponto.csv e metadados.json.
"""
import argparse
import json
from pathlib import Path

from algamar_ml import config, pipeline
from algamar_ml.features.horizonte import texto_mes
from algamar_ml.modelos import treino as tr
from algamar_ml.previsao import mes_de_emissao, pico_por_ponto, preve, resumo_regioes

ap = argparse.ArgumentParser()
ap.add_argument("--versao", default=None, help="padrão: [modelo].oficial do config/ml.toml")
args = ap.parse_args()

cfg = config.carrega()
args.versao = args.versao or cfg["modelo"]["oficial"]
pasta_modelo = Path(cfg["caminhos"]["modelos"]) / args.versao
modelo, cartao = tr.carrega(pasta_modelo)
cfg["features"].update(cartao["flags"])

cubo, painel = pipeline.carrega_cubo(cfg)
matrizes, _ = pipeline.fisica(cfg, cubo)
t0 = mes_de_emissao(cubo)
ano, mes = divmod(t0, 12)
fontes_t0 = painel[(painel.ano == ano) & (painel.mes == mes + 1)][["fonte_clorofila", "fonte_temperatura", "fonte_salinidade"]].iloc[0]
fontes = f"CHL={fontes_t0['fonte_clorofila']}; SST={fontes_t0['fonte_temperatura']}; SSS={fontes_t0['fonte_salinidade']}"

previsao = preve(modelo, cartao, cubo, cfg, fisica=matrizes, t0=t0, fontes=fontes)
pasta = Path(cfg["caminhos"]["previsoes"]) / str(texto_mes(t0))
pasta.mkdir(parents=True, exist_ok=True)
previsao.to_parquet(pasta / "previsao_pontos.parquet", index=False)
previsao.to_csv(pasta / "previsao_pontos.csv", index=False)
regioes = resumo_regioes(previsao)
regioes.to_csv(pasta / "resumo_regioes.csv", index=False)
pico_por_ponto(previsao).to_csv(pasta / "pico_por_ponto.csv", index=False)
meta = {"issue_month": str(texto_mes(t0)), "meses_previstos": sorted(previsao["target_date"].unique().tolist()),
        "model_version": cartao["versao"], "status_modelo": cartao["status"], "fontes_mes_emissao": fontes,
        "pontos": int(previsao["point_id"].nunique()),
        "pontos_com_estado_incompleto": int((~previsao.groupby("point_id")["estado_t0_completo"].first()).sum()),
        "limiar_alto": cartao["limiar_alto"], "limiar_moderado": cartao["limiar_moderado"],
        "campos_nao_disponiveis": ["município", "estação de coleta", "espécie/toxicidade"]}
(pasta / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

print(json.dumps(meta, ensure_ascii=False, indent=1))
tabela = regioes.pivot(index="target_date", columns="region", values="prob_media").round(3)
print("\nProbabilidade média por região e mês:\n", tabela.to_string())
print(f"\nSalvo em {pasta}")
