"""Etapa 13.6 (E7) — GLORYS12 mensal (reanálise) na caixa do ML, 2000-01 a 2024-12: T, S, u, v de superfície e MLD.

Controle "física de reanálise" para o experimento E0–E7: o CROCO é forçado pelo GLORYS, então só um ganho do CROCO acima
deste controle é atribuível ao CROCO. Produto GLOBAL_MULTIYEAR_PHY_001_030 (cmems_mod_glo_phy_my_0.083deg_P1M-m), MY.
Login: credenciais que o próprio usuário configurou (nunca lidas aqui).
Saída: dados/brutos/glorys_mensal/glorys_mensal_2000_2024.nc + proveniencia.json
"""
import datetime as dt
import json
from pathlib import Path

import copernicusmarine as cm

D = Path("dados/brutos/glorys_mensal"); D.mkdir(parents=True, exist_ok=True)
kw = dict(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m", variables=["thetao", "so", "uo", "vo", "mlotst"],
          minimum_longitude=-47.25, maximum_longitude=-44.55, minimum_latitude=-25.55, maximum_latitude=-23.05,
          minimum_depth=0.0, maximum_depth=1.0, start_datetime="2000-01-01T00:00:00", end_datetime="2024-12-01T00:00:00",
          output_directory=str(D), output_filename="glorys_mensal_2000_2024.nc", overwrite=True, disable_progress_bar=True)
acesso = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
r = cm.subset(**kw)
prov = {"provider": "Copernicus Marine Service", "product_id": "GLOBAL_MULTIYEAR_PHY_001_030", "classe": "REANALISE",
        "comando": "copernicusmarine.subset(" + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")",
        "acesso_utc": acesso, "extensao": {e.coordinate_id: [str(e.minimum), str(e.maximum)] for e in r.coordinates_extent},
        "credenciais": "não registradas (login do próprio usuário)"}
(D / "proveniencia.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(prov, ensure_ascii=False, indent=1))
