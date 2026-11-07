"""Baixa as condições RECENTES (sistema operacional / NRT do Copernicus) para completar a reanálise de
baixar_condicoes_regiao.py, que chega com alguns meses de atraso.

| produto (Copernicus Marine)                        | variáveis                  | período                           |
|----------------------------------------------------|----------------------------|-----------------------------------|
| cmems_mod_glo_phy-thetao/so_anfc_0.083deg_P1M-m    | thetao, so (superfície)    | 2022-06 em diante (mensal)        |
| cmems_mod_glo_phy_anfc_0.083deg_P1M-m              | mlotst                     | 2022-06 em diante (mensal)        |
| cmems_mod_glo_phy-thetao/so_anfc_0.083deg_P1D-m    | thetao, so (superfície)    | mês corrente (diário, inclui os   |
| cmems_mod_glo_phy_anfc_0.083deg_P1D-m              | mlotst                     |  10 dias de previsão do sistema)  |
| cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H        | vento leste/norte          | meses que faltam na reanálise     |

O período em comum com a GLORYS (2022-06 até o fim dela) serve para corrigir a diferença entre os dois
sistemas (ver indice_condicoes.py). Uso: python baixar_condicoes_nrt.py   (refaz tudo; os arquivos são pequenos)
Saída: dados/condicoes/nrt_*.nc
"""
import datetime as dt
import json
from pathlib import Path

import copernicusmarine as cm
import xarray as xr

from algamar_ml import config

CAIXA = dict(minimum_longitude=-49.5, maximum_longitude=-40.5, minimum_latitude=-29.5, maximum_latitude=-22.0)
SUPERFICIE = {"minimum_depth": 0.0, "maximum_depth": 1.0}

destino = Path(config.carrega()["caminhos"]["painel"]).parent.parent / "condicoes"
hoje = dt.date.today()
inicio_mes = hoje.replace(day=1)
fim_mes = (inicio_mes + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)


def baixa(nome, ds, variaveis, inicio, fim, extra=None):
    arq = destino / f"{nome}.nc"
    arq.unlink(missing_ok=True)
    print(f"baixando {ds} {inicio}..{fim} ...", flush=True)
    cm.subset(dataset_id=ds, variables=variaveis, start_datetime=f"{inicio}T00:00:00", end_datetime=f"{fim}T23:59:59",
              output_directory=str(destino), output_filename=arq.name, disable_progress_bar=True,
              coordinates_selection_method="nearest", **CAIXA, **(extra or {}))
    return {"dataset_id": ds, "variaveis": variaveis, "inicio": str(inicio), "fim": str(fim),
            "mb": round(arq.stat().st_size / 1e6, 1)}


prov = {}
for sufixo, ds, var, extra in [("thetao", "cmems_mod_glo_phy-thetao_anfc_0.083deg", "thetao", SUPERFICIE),
                               ("so", "cmems_mod_glo_phy-so_anfc_0.083deg", "so", SUPERFICIE),
                               ("mlotst", "cmems_mod_glo_phy_anfc_0.083deg", "mlotst", None)]:
    prov[f"nrt_{sufixo}_mensal"] = baixa(f"nrt_{sufixo}_mensal", ds + "_P1M-m", [var], "2022-06-01", hoje, extra)
    prov[f"nrt_{sufixo}_diario"] = baixa(f"nrt_{sufixo}_diario", ds + "_P1D-m", [var], inicio_mes, fim_mes, extra)

# vento: só os meses depois do fim da reanálise mensal (o horário é pesado); média mensal já aqui
w_my = xr.open_dataset(destino / "vento_mensal.nc")
ini_vento = (dt.date.fromisoformat(str(w_my.time.values[-1])[:10]) + dt.timedelta(days=32)).replace(day=1)
prov["nrt_vento_horario"] = baixa("nrt_vento_horario", "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
                                  ["eastward_wind", "northward_wind"], ini_vento, hoje)
h = xr.open_dataset(destino / "nrt_vento_horario.nc")
h.resample(time="MS").mean().to_netcdf(destino / "nrt_vento_mensal.nc")
h.close()
(destino / "nrt_vento_horario.nc").unlink()

for v in prov.values():
    v["baixado_em"] = dt.datetime.now(dt.UTC).isoformat()
(destino / "proveniencia_nrt.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
print("ok")
