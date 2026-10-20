"""Etapa 13.4 — pipeline operacional (HISTÓRICO + APIs; sem Navier–Stokes):

  api_ingestion → validation → temporal_alignment → spatial_alignment → feature_generation → forecast

Uso:
  python pipeline_operacional.py --versao baseline-hist-api-v1            (usa os brutos já baixados)
  python pipeline_operacional.py --versao baseline-hist-api-v1 --baixar   (consulta o catálogo, baixa meses novos e refaz o painel)
  python pipeline_operacional.py --versao ... --hoje 2026-10-09           (fixa a "data de hoje" para teste/reprodução)
Saída: previsoes/operacional_<mês de emissão>/ (previsao_pontos.parquet/csv, resumo_regioes.csv,
       relatorio_operacional.json com validação, fallback e proveniência por feature).
Nada é inventado: falhas viram NaN com flag, recuo do mês de emissão ou "SEM_EMISSAO".
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from algamar_ml import config, operacional as op, pipeline
from algamar_ml.apis.copernicus import PRODUTOS, cobertura
from algamar_ml.features.horizonte import cubo_do_painel, texto_mes
from algamar_ml.modelos import treino as tr
from algamar_ml.previsao import preve, resumo_regioes

ap = argparse.ArgumentParser()
ap.add_argument("--versao", default=None, help="padrão: [modelo].oficial do config/ml.toml")
ap.add_argument("--baixar", action="store_true")
ap.add_argument("--hoje", default=None)
args = ap.parse_args()
hoje = dt.date.fromisoformat(args.hoje) if args.hoje else dt.date.today()
cfg = config.carrega()
args.versao = args.versao or cfg["modelo"]["oficial"]
rel = {"hoje": str(hoje), "versao_modelo": args.versao, "etapas": {}}

# ------------------------------------------------------------------ 1. api_ingestion
ing = {"catalogo": {}, "download": "não solicitado (--baixar ausente): usa os brutos já baixados"}
for ps in PRODUTOS.values():
    for p in ps:
        try:
            a, b = cobertura(p.dataset_id)
            ing["catalogo"][p.dataset_id] = {"tipo": p.tipo, "de": str(a), "ate": str(b), "status": "OK"}
        except Exception as e:  # API indisponível: registra e segue com os dados locais
            ing["catalogo"][p.dataset_id] = {"tipo": p.tipo, "status": "INDISPONIVEL", "erro": f"{type(e).__name__}: {str(e)[:200]}"}
if args.baixar:
    r1 = subprocess.run([sys.executable, "atualizar_dados_atuais.py"], capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, "construir_painel.py"], capture_output=True, text=True)
    ing["download"] = {"atualizar_exit": r1.returncode, "painel_exit": r2.returncode, "saida": r1.stdout[-2000:]}
rel["etapas"]["api_ingestion"] = ing

# ------------------------------------------------------------------ 2. validation (arquivos diários dos meses novos)
painel = pd.read_parquet(cfg["caminhos"]["painel"])
grade = painel[["latitude", "longitude"]].drop_duplicates()
val_brutos = []
for reg in sorted(Path(cfg["caminhos"]["brutos"]).glob("*/*/*.json")):
    meta = json.loads(reg.read_text(encoding="utf-8"))
    nc = reg.with_suffix(".nc")
    if not nc.exists():
        val_brutos.append({**meta, "status": "FALHA", "motivo": "arquivo .nc ausente"})
        continue
    with xr.open_dataset(nc) as ds:
        v = op.valida_tempos_diarios(ds["time"].values, meta["ano"], meta["mes"])
    val_brutos.append({"variavel": reg.parent.parent.name, "dataset_id": meta["dataset_id"], "tipo": meta["tipo"],
                       "ano": meta["ano"], "mes": meta["mes"], "status": v["status"], "cobertura_dias": round(v["cobertura"], 3),
                       "checagens": v["checagens"]})
rel["etapas"]["validation"] = {"arquivos": len(val_brutos), "por_status": pd.Series([x["status"] for x in val_brutos]).value_counts().to_dict(),
                               "falhas": [x for x in val_brutos if x["status"] != "OK"]}

# ------------------------------------------------------------------ 3. temporal_alignment (mês civil UTC; mês sem cobertura → NaN)
col = {"clorofila": ["clorofila_media", "clorofila_maxima"], "temperatura": ["temperatura_media"], "salinidade": ["salinidade_media"]}
anulados = []
usados = painel[["ano", "mes", "fonte_clorofila", "fonte_temperatura", "fonte_salinidade"]].drop_duplicates(["ano", "mes"])
for x in val_brutos:
    if x["status"] == "FALHA":
        fonte_usada = usados[(usados.ano == x["ano"]) & (usados.mes == x["mes"])][f"fonte_{x['variavel']}"]
        if len(fonte_usada) and x["dataset_id"] in str(fonte_usada.iloc[0]):
            sel = (painel.ano == x["ano"]) & (painel.mes == x["mes"])
            painel.loc[sel, col[x["variavel"]]] = np.nan
            painel.loc[sel, f"fonte_{x['variavel']}"] = "ANULADO(validação temporal)"
            anulados.append(f"{x['variavel']} {x['ano']}-{x['mes']:02d}")
rel["etapas"]["temporal_alignment"] = {"regra": "mês civil UTC; médias mensais de dias UTC; arquivo com < 80 % dos dias ou tempo inválido → variável NaN no mês",
                                       "variaveis_anuladas": anulados}

# ------------------------------------------------------------------ 4. spatial_alignment + decisão da emissão (fallback)
modelo, cartao = tr.carrega(Path(cfg["caminhos"]["modelos"]) / args.versao)
cfg["features"].update(cartao["flags"])
vars_modelo = ["clorofila", "temperatura"] + (["salinidade"] if cartao["flags"].get("use_salinity") else [])
emissao = op.decide_emissao(painel, grade, cfg, hoje, vars_modelo)
rel["etapas"]["spatial_alignment"] = {"grade": "CHL L4 4 km, 1 790 pontos do histórico; T e S interpoladas (bilinear) na grade CHL",
                                      "validacao_t0": emissao.get("validacao_t0", {}).get("checagens")}
rel["emissao"] = {k: v for k, v in emissao.items() if k != "validacao_t0"}
if emissao["t0"] is None:
    print(json.dumps(rel, ensure_ascii=False, indent=1, default=str))
    raise SystemExit("SEM_EMISSAO: " + emissao["motivo"])
t0 = emissao["t0"]

# ------------------------------------------------------------------ 5. feature_generation (+ proveniência) e 6. forecast
cubo = cubo_do_painel(painel)
fontes_t0 = painel[(painel.ano * 12 + painel.mes - 1) == t0][["fonte_clorofila", "fonte_temperatura", "fonte_salinidade"]].iloc[0]
fontes = f"CHL={fontes_t0.fonte_clorofila}; SST={fontes_t0.fonte_temperatura}; SSS={fontes_t0.fonte_salinidade}"
prev = preve(modelo, cartao, cubo, cfg, t0=t0, fontes=fontes)
prov = op.proveniencia_features(painel, cartao["features"], t0)
assert all(p.get("todos_ate_t0", True) for p in prov.values()), "feature usando mês posterior a t0"
regs = emissao["validacao_t0"].get("regioes", {})
prev["regiao_status"] = prev["region"].map(lambda r: regs.get(r, {}).get("status", "?"))
prev["emissao_status"] = emissao["status"]
prev["atraso_meses"] = emissao["atraso_meses"]
rel["etapas"]["feature_generation"] = {"features": cartao["features"], "proveniencia": prov}
pasta = Path(cfg["caminhos"]["previsoes"]) / f"operacional_{texto_mes(t0)}"
pasta.mkdir(parents=True, exist_ok=True)
prev.to_parquet(pasta / "previsao_pontos.parquet", index=False)
prev.to_csv(pasta / "previsao_pontos.csv", index=False)
resumo_regioes(prev).to_csv(pasta / "resumo_regioes.csv", index=False)
rel["etapas"]["forecast"] = {"pasta": str(pasta), "linhas": int(len(prev)), "meses_previstos": sorted(prev.target_date.unique().tolist()),
                             "pontos_estado_incompleto": int((~prev.groupby("point_id")["estado_t0_completo"].first()).sum())}
(pasta / "relatorio_operacional.json").write_text(json.dumps(rel, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in rel.items() if k != "etapas"}, ensure_ascii=False, indent=1, default=str))
print("validação dos brutos:", rel["etapas"]["validation"]["por_status"], "| anulados:", anulados)
print("forecast:", rel["etapas"]["forecast"])
