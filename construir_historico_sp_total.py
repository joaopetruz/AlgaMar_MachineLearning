"""Histórico mensal 2000–2024 do litoral de SP inteiro, a partir dos anuais de baixar_litoral_sp_total.py.

Mesmo método da equipe (algamar_ml/processamento/mensal.py: SST/SSS bilineares na grade CHL de 4 km, médias e máximo
do mês civil). Diferenças em relação ao CSV da equipe:
- pontos: caixa da equipe + extensão sul até a divisa SP/PR (espacial.dentro_sp);
- salinidade: SEM preenchimento por vizinho (o CSV da equipe preenche a costa até 55 km); o modelo não usa SSS.
Ponto oceânico = CHL com dado em algum dia do período (terra é NaN no produto L4).

Uso:  python construir_historico_sp_total.py [--anos 2000-2024]
Saída: [caminhos].historico_equipe do config (CSV no formato do CSV da equipe) + .manifesto.json, e uma conferência
contra o CSV da equipe nos pontos em comum (resultados/sp_total/conferencia_equipe.json).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml import config
from algamar_ml.espacial import dentro_sp
from algamar_ml.processamento.mensal import abre, processa
from algamar_ml.processamento.painel import sha256

ap = argparse.ArgumentParser()
ap.add_argument("--anos", default="2000-2024")
args = ap.parse_args()
a0, a1 = (int(x) for x in args.anos.split("-"))
cfg = config.carrega()
raiz = Path(cfg["caminhos"]["brutos_historicos"])
COLS = ["latitude", "longitude", "ano", "mes", "clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"]


def arquivo(var, ano):
    achados = list((raiz / var).glob(f"*/{var}_{ano}.nc"))
    return achados[0] if achados else None


partes, entradas = [], []
for ano in range(a0, a1 + 1):
    arqs = {v: arquivo(v, ano) for v in ("clorofila", "temperatura", "salinidade")}
    if arqs["clorofila"] is None:
        raise SystemExit(f"falta clorofila {ano}: rode baixar_litoral_sp_total.py")
    ds = {v: abre([a]) if a else None for v, a in arqs.items()}
    t = processa(ds["clorofila"], ds["temperatura"], ds["salinidade"])
    t = t[dentro_sp(t.latitude, t.longitude, cfg["area"])]
    partes.append(t[COLS])
    entradas += [{"variavel": v, "ano": ano, "arquivo": str(a), "sha256": sha256(a)} for v, a in arqs.items() if a]
    print(ano, len(t), flush=True)

hist = pd.concat(partes, ignore_index=True)
oceano = hist.groupby(["latitude", "longitude"]).clorofila_media.transform(lambda s: s.notna().any())
hist = hist[oceano].sort_values(["latitude", "longitude", "ano", "mes"]).reset_index(drop=True)
destino = Path(cfg["caminhos"]["historico_equipe"])
hist.to_csv(destino, index=False)
pontos = hist[["latitude", "longitude"]].drop_duplicates()
man = {"arquivo": str(destino), "sha256": sha256(destino), "linhas": int(len(hist)), "pontos": int(len(pontos)),
       "periodo": [f"{a0}-01", f"{a1}-12"], "area": cfg["area"], "entradas": entradas,
       "nan_por_coluna": {c: int(hist[c].isna().sum()) for c in COLS[4:]}}
destino.with_suffix(".manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")

# conferência contra o CSV da equipe (mesmos pontos e meses)
equipe_csv = config.RAIZ / "dados" / "dados_mensais_2000_2024_sp_com_salinidade.csv"
eq = pd.read_csv(equipe_csv)
for c in ("latitude", "longitude"):
    eq[c] = eq[c].round(5)
j = eq.merge(hist, on=["latitude", "longitude", "ano", "mes"], suffixes=("_equipe", "_novo"))
conf = {"pontos_equipe": int(eq[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "pontos_equipe_no_novo": int(j[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "pontos_novo": int(len(pontos)), "linhas_comparadas": int(len(j))}
for c in COLS[4:]:
    d = (j[f"{c}_novo"] - j[f"{c}_equipe"]).abs()
    conf[c] = {"dif_abs_mediana": float(d.median()), "dif_abs_p99": float(d.quantile(0.99)), "dif_abs_max": float(d.max()),
               "iguais_1e-4": float((d < 1e-4).mean()), "nan_novo_com_equipe_valido": int((j[f"{c}_novo"].isna() & j[f"{c}_equipe"].notna()).sum())}
o = Path("resultados/sp_total"); o.mkdir(parents=True, exist_ok=True)
(o / "conferencia_equipe.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: man[k] for k in ("linhas", "pontos", "periodo")}, ensure_ascii=False))
print(json.dumps(conf, ensure_ascii=False, indent=1))
