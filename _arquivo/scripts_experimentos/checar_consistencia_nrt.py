"""Compara, em 2024, os produtos NRT (usados para o estado atual) com os MY/REP do histórico.

Se o NRT tiver viés em relação ao MY/REP, o modelo treinado no histórico veria um "estado
atual" deslocado. Saída: resultados/consistencia_nrt_2024.json (viés, RMSE e correlação por
variável, na grade CHL e nos meses de 2024).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml import config
from algamar_ml.processamento.mensal import abre, processa
from algamar_ml.processamento.painel import le_historico

cfg = config.carrega()
brutos = Path(cfg["caminhos"]["brutos"])
hist = le_historico(cfg["caminhos"]["historico_equipe"])
hist = hist[hist.ano == 2024]
equipe = Path(__file__).resolve().parents[2] / "cfd_navier_stokes" / "data" / "raw" / "ml_equipe"
chl_2024 = abre([equipe / "clorofila_2024_sp.nc"])

resultado = {}
for variavel, coluna, pasta in (("temperatura", "temperatura_media", "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"),
                                ("salinidade", "salinidade_media", "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D")):
    arquivos = sorted((brutos / variavel / pasta).glob(f"{variavel}_2024-*.nc"))
    if not arquivos:
        resultado[variavel] = "sem arquivos NRT de 2024"
        continue
    ds = abre(arquivos)
    t = processa(chl_2024, ds if variavel == "temperatura" else None, ds if variavel == "salinidade" else None)
    m = hist.merge(t[["latitude", "longitude", "ano", "mes", coluna]], on=["latitude", "longitude", "ano", "mes"],
                   suffixes=("_my", "_nrt")).dropna(subset=[coluna + "_my", coluna + "_nrt"])
    d = m[coluna + "_nrt"] - m[coluna + "_my"]
    meses_nrt = sorted({int(a.stem[-2:]) for a in arquivos})
    resultado[variavel] = {
        "produto_nrt": pasta, "meses_2024_comparados": meses_nrt, "n": int(len(m)),
        "vies_nrt_menos_my": float(d.mean()), "rmse": float(np.sqrt((d ** 2).mean())),
        "correlacao": float(np.corrcoef(m[coluna + "_nrt"], m[coluna + "_my"])[0, 1]),
        "vies_por_mes": {int(k): float(v) for k, v in d.groupby(m["mes"]).mean().items()},
    }

Path("resultados").mkdir(exist_ok=True)
Path("resultados/consistencia_nrt_2024.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(resultado, ensure_ascii=False, indent=1))
