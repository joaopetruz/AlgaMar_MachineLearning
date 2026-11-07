"""AlgaMar — ponto de entrada único do pipeline final (Etapa 13.8). Tudo vem de config/ml.toml.

  python algamar.py previsao                          fluxo completo com o modelo oficial e a data de hoje
  python algamar.py previsao --hoje 2026-10-09        data fixa (reprodução)
  python algamar.py previsao --versao X --baixar      outro modelo registrado; baixa meses novos antes
  python algamar.py treinar --versao NOVA             treina e registra uma versão NOVA (nunca sobrescreve)
  python algamar.py treinar --versao NOVA --ensemble  idem, como ensemble de ordens ([ensemble] do config; 14.1)
  python algamar.py modelos                           lista o registro de modelos
  python algamar.py --config outro.toml ...           outra configuração

Saídas: saidas/<run_id>/{previsao.parquet, risco_regioes.parquet, previsao.json} e saidas/ULTIMA.json.
Logs: logs/execucoes/<run_id>.json e logs/execucoes/execucoes.jsonl.
Código de saída: 0 OK/ATRASADO, 10 DESATUALIZADO, 20 DADOS_INSUFICIENTES, 30 ERRO_DADOS, 40 ERRO_MODELO, 50 ERRO.
"""
import argparse
import datetime as dt
import json
import sys

from algamar_ml import config, fluxo, registro

ap = argparse.ArgumentParser(description="AlgaMar — pipeline final")
ap.add_argument("--config", default=None)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("previsao"); p.add_argument("--versao"); p.add_argument("--hoje"); p.add_argument("--baixar", action="store_true")
t = sub.add_parser("treinar"); t.add_argument("--versao", required=True); t.add_argument("--ensemble", action="store_true")
sub.add_parser("modelos")
a = ap.parse_args()
cfg = config.carrega(a.config)

if a.cmd == "modelos":
    print(json.dumps(registro.lista(cfg["caminhos"]["modelos"]), ensure_ascii=False, indent=1)); sys.exit(0)
if a.cmd == "treinar":
    if a.ensemble:
        cfg.setdefault("ensemble", {"k": 8, "seed_mestra": 42, "calibracao": ""})["ativo"] = True
    status, log = fluxo.executa_treino(cfg, a.versao, a.config)
else:
    status, log = fluxo.executa_previsao(cfg, a.versao, dt.date.fromisoformat(a.hoje) if a.hoje else None, a.baixar, config_arquivo=a.config)
print(json.dumps({k: log[k] for k in ("run_id", "status", "periodo", "resultado", "avisos", "erros")}, ensure_ascii=False, indent=1, default=str))
sys.exit(fluxo.CODIGO_SAIDA.get(status, 50))
