"""Baixa as condições ambientais MENSAIS do litoral de Cabo Frio (RJ) ao sul de SC, para o índice de
condições favoráveis às espécies tóxicas (cruza com os registros do OBIS em dados/obis/).

| produto (Copernicus Marine)                          | variáveis                                   | resolução |
|------------------------------------------------------|---------------------------------------------|-----------|
| cmems_mod_glo_phy_my_0.083deg_P1M-m  (GLORYS12)      | thetao, so (superfície), mlotst, uo, vo     | 1/12°     |
| cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M    | CHL, DINO (dinoflagelados), DIATO (diatom.) | 4 km      |
| cmems_obs-wind_glo_phy_my_l4_P1M                     | vento e tensão do vento (leste/norte)       | 0,125°    |

Uso (login Copernicus já configurado):  python baixar_condicoes_regiao.py  [--inicio 2000-01]
Saída: dados/condicoes/<produto>.nc + proveniencia.json. Pula o que já existe (apague para baixar de novo).
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import copernicusmarine as cm

from algamar_ml import config

CAIXA = dict(minimum_longitude=-49.5, maximum_longitude=-40.5, minimum_latitude=-29.5, maximum_latitude=-22.0)
PRODUTOS = [
    ("glorys_mensal", "cmems_mod_glo_phy_my_0.083deg_P1M-m", ["thetao", "so", "mlotst", "uo", "vo"],
     {"minimum_depth": 0.0, "maximum_depth": 1.0}),
    ("clorofila_grupos_mensal", "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M", ["CHL", "DINO", "DIATO"], {}),
    ("vento_mensal", "cmems_obs-wind_glo_phy_my_l4_P1M",
     ["eastward_wind", "northward_wind", "eastward_stress", "northward_stress"], {}),
]

ap = argparse.ArgumentParser()
ap.add_argument("--inicio", default="2000-01")
a = ap.parse_args()

destino = Path(config.carrega()["caminhos"]["painel"]).parent.parent / "condicoes"
destino.mkdir(parents=True, exist_ok=True)
prov = {}
for nome, ds, variaveis, extra in PRODUTOS:
    arq = destino / f"{nome}.nc"
    if not arq.exists():
        print(f"baixando {ds} ...", flush=True)
        cm.subset(dataset_id=ds, variables=variaveis, start_datetime=f"{a.inicio}-01T00:00:00",
                  end_datetime=dt.date.today().isoformat() + "T00:00:00", output_directory=str(destino),
                  output_filename=arq.name, disable_progress_bar=True, coordinates_selection_method="nearest",
                  **CAIXA, **extra)
    prov[nome] = {"dataset_id": ds, "variaveis": variaveis, "caixa": CAIXA, "inicio": a.inicio,
                  "arquivo": arq.name, "mb": round(arq.stat().st_size / 1e6, 1),
                  "baixado_em": dt.datetime.now(dt.UTC).isoformat()}
    print(f"  {arq.name}: {prov[nome]['mb']} MB", flush=True)
(destino / "proveniencia.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
