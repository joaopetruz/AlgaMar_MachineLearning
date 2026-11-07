import numpy as np
import pandas as pd
import pytest

from algamar_ml.croco import features_do_conjunto, junta_croco, le_croco_mensal

PONTOS = [(-24.000001, -46.0, "D2_VALIDO"), (-24.5, -45.5, "NAO_VALIDADO_COSTEIRO"), (-23.9, -46.3, "NAO_USAR_ESTUARIO_CANAL")]


def croco_sintetico(anos=range(2016, 2019)):
    linhas = []
    for lat, lon, flag in PONTOS:
        for ano in anos:
            for mes in range(1, 13):
                linhas.append(dict(latitude=lat, longitude=lon, ano=ano, mes=mes, validade_espacial=flag,
                                   temperature_surface_mean=20 + mes + (ano - 2016) * 100,
                                   salinity_surface_mean=35.0, mld_mean=10.0, n2_max_mean=1e-4))
    return pd.DataFrame(linhas)


def ml_sintetico():
    linhas = [dict(latitude=round(lat, 6), longitude=lon, ano=2018, mes=m, floracao=0)
              for lat, lon, _ in PONTOS for m in range(1, 13)]
    return pd.DataFrame(linhas)


def test_nomes_por_modo():
    assert features_do_conjunto("CROCO_TS", "NOWCAST") == ["temperature_surface_mean", "salinity_surface_mean"]
    assert features_do_conjunto("CROCO_TS") == ["temperature_surface_mean__lag12", "salinity_surface_mean__lag12"]
    with pytest.raises(ValueError):
        features_do_conjunto("CROCO_TS", "QUALQUER")


def test_join_sazonal_pega_o_ano_anterior_por_data(tmp_path):
    caminho = tmp_path / "croco.parquet"
    croco_sintetico().to_parquet(caminho)
    tabela, rel = junta_croco(ml_sintetico(), le_croco_mensal(caminho), "CROCO_TS", "FORECAST_SAZONAL")
    assert rel["linhas_NAO_USAR_removidas"] == 12 and rel["pontos_NAO_USAR"] == 1
    assert rel["ml_sem_croco"] == 0 and len(tabela) == 24
    jul = tabela[(tabela.latitude == -24.0) & (tabela.mes == 7)].iloc[0]
    assert jul["temperature_surface_mean__lag12"] == 20 + 7 + 100  # 2017-07, não 2018-07
    assert "validade_espacial" in tabela.columns


def test_lag_sem_mes_anterior_vira_nan(tmp_path):
    croco = croco_sintetico(anos=[2018])
    tabela, rel = junta_croco(ml_sintetico(), croco, "CROCO_TS", "FORECAST_SAZONAL")
    assert tabela["temperature_surface_mean__lag12"].isna().all()
    assert rel["linhas_com_feature_croco_completa"] == 0


def test_chave_repetida_no_croco_e_recusada(tmp_path):
    croco = croco_sintetico()
    croco = pd.concat([croco, croco.iloc[:1]])
    caminho = tmp_path / "croco.csv"
    croco.to_csv(caminho, index=False)
    with pytest.raises(ValueError):
        le_croco_mensal(caminho)


def test_coluna_ausente_e_recusada():
    croco = croco_sintetico().drop(columns="n2_max_mean")
    with pytest.raises(ValueError):
        junta_croco(ml_sintetico(), croco, "CROCO_CORE", "NOWCAST", permite_nowcast=True)
