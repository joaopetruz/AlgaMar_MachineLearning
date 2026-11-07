import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from algamar_ml import api_ambiente


@pytest.fixture()
def tabela(monkeypatch):
    linhas = []
    for mun in ("Santos", "Ubatuba"):
        for mes in pd.period_range("2023-01", "2026-09", freq="M"):
            linhas.append(dict(municipio=mun, mes=mes, temperatura_c=20.0 + (mes.year == 2026), salinidade_psu=35.0,
                               clorofila_mg_m3=1.0 if mun == "Santos" else 3.0, clorofila_max_mg_m3=2.0,
                               corrente_u=0.0 if mes <= pd.Period("2026-08", "M") else np.nan, corrente_v=-0.1,
                               ph=8.1 if mes.year == 2026 else np.nan))
    monkeypatch.setattr(api_ambiente, "tabela", lambda: pd.DataFrame(linhas))


def test_meses_passados_medidos_e_futuros_com_normal(tabela):
    r = api_ambiente.monta(meses_passados=9, meses=3, hoje=dt.date(2026, 10, 27))
    assert [m["mes"] for m in r["meses"]][0] == "2026-01" and r["meses"][-1]["mes"] == "2027-01" and len(r["meses"]) == 13
    set_, out = r["meses"][8]["variaveis"], r["meses"][9]["variaveis"]
    assert set_["temperatura_c"] == {"valor": 21.0, "fonte": "medido"} and set_["clorofila_mg_m3"]["valor"] == 2.0
    assert out["temperatura_c"] == {"valor": 20.0, "fonte": "normal"} and r["meses"][9]["futuro"]   # normal 2000–2024
    assert out["ph"] == {"valor": 8.1, "fonte": "ultimo"}
    assert set_["corrente_kmh"] == {"valor": 0.36, "fonte": "normal", "direcao": "S"}               # GLORYS acaba em ago


def test_regiao_e_municipio(tabela):
    assert api_ambiente.monta("Santos", hoje=dt.date(2026, 10, 27))["meses"][0]["variaveis"]["clorofila_mg_m3"]["valor"] == 1.0
    assert api_ambiente.monta("Litoral Norte", hoje=dt.date(2026, 10, 27))["locais"] == ["Ubatuba", "Caraguatatuba", "Ilhabela", "São Sebastião"]
    with pytest.raises(HTTPException):
        api_ambiente.monta("Paraná")
