"""Baixa CHL, SST e SSS diárias (MY/REP) por ANO na caixa ampliada do litoral de SP (config [area]), 2000–2024.

Motivo (2026-10-09): a caixa herdada da equipe (lon −47,0…−44,8) deixava Iguape, Ilha Comprida e Cananéia fora do ML.
Os meses de 2025 em diante vêm do fluxo normal (atualizar_dados_atuais.py), que usa a mesma caixa do config.

Uso:  python baixar_litoral_sp_total.py [--anos 2000-2024] [--variaveis clorofila,temperatura,salinidade]
Saída: <brutos_historicos>/<variavel>/<dataset_id>/<variavel>_<ano>.nc + .json (proveniência). Pula o que já existe.
Login: o do próprio usuário (copernicusmarine login); este script nunca lê credenciais.
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import copernicusmarine as cm

from algamar_ml import config
from algamar_ml.apis.copernicus import PRODUTOS, caixa

ap = argparse.ArgumentParser()
ap.add_argument("--anos", default="2000-2024")
ap.add_argument("--variaveis", default="clorofila,temperatura,salinidade")
args = ap.parse_args()
a0, a1 = (int(x) for x in args.anos.split("-"))
cfg = config.carrega()
raiz = Path(cfg["caminhos"]["brutos_historicos"])

for ano in range(a0, a1 + 1):
    for var in args.variaveis.split(","):
        p = next(p for p in PRODUTOS[var] if p.tipo == "MY")
        destino = raiz / var / p.dataset_id; destino.mkdir(parents=True, exist_ok=True)
        nome = f"{var}_{ano}.nc"
        if (destino / nome).exists():
            continue
        t0 = time.time()
        extra = {"minimum_depth": 0.0, "maximum_depth": 1.0} if var == "salinidade" else {}
        cm.subset(dataset_id=p.dataset_id, variables=[p.variavel_origem], start_datetime=f"{ano}-01-01T00:00:00",
                  end_datetime=f"{ano}-12-31T00:00:00", output_directory=str(destino), output_filename=nome,
                  disable_progress_bar=True, **caixa(cfg, var), **extra)
        (destino / nome.replace(".nc", ".json")).write_text(json.dumps(
            {"dataset_id": p.dataset_id, "tipo": "MY", "variavel": p.variavel_origem, "ano": ano, "caixa": caixa(cfg, var),
             "baixado_em": dt.datetime.now(dt.UTC).isoformat()}, indent=1), encoding="utf-8")
        print(f"{var} {ano} {(destino / nome).stat().st_size / 1e6:.1f} MB {time.time() - t0:.0f}s", flush=True)
print("ok")
