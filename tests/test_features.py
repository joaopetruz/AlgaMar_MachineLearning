import numpy as np
import pandas as pd
import pytest

from algamar_ml.features import (
    climatologia_passada, constroi_features, lag_por_data, media_movel_passada,
)


def mensal_sintetico(anos=range(2000, 2006), pontos=((-24.0, -46.0), (-24.5, -45.5)), seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for lat, lon in pontos:
        for ano in anos:
            for mes in range(1, 13):
                chl = rng.gamma(2.0, 0.5)
                linhas.append(dict(latitude=lat, longitude=lon, ano=ano, mes=mes,
                                   clorofila_media=chl, clorofila_maxima=chl * 1.8,
                                   temperatura_media=22 + rng.normal(), salinidade_media=35 + rng.normal(0, 0.3)))
    return pd.DataFrame(linhas)


def test_lag_por_data_respeita_lacuna():
    m = mensal_sintetico()
    m = m.drop(m[(m.latitude == -24.0) & (m.ano == 2002) & (m.mes == 5)].index).reset_index(drop=True)
    m["lag"] = lag_por_data(m, "temperatura_media", 12)
    linha = m[(m.latitude == -24.0) & (m.ano == 2003) & (m.mes == 5)]
    assert linha["lag"].isna().all()  # o mês de 12 meses antes não existe: NaN, nunca outro mês
    linha = m[(m.latitude == -24.0) & (m.ano == 2003) & (m.mes == 6)].iloc[0]
    esperado = m[(m.latitude == -24.0) & (m.ano == 2002) & (m.mes == 6)]["temperatura_media"].iloc[0]
    assert linha["lag"] == esperado


def test_climatologia_so_usa_anos_anteriores():
    m = mensal_sintetico()
    m["clim"] = climatologia_passada(m, "clorofila_media")
    sel = (m.latitude == -24.0) & (m.mes == 3)
    serie = m[sel].sort_values("ano")
    assert np.isnan(serie["clim"].iloc[0])
    for i in range(1, len(serie)):
        assert serie["clim"].iloc[i] == pytest.approx(serie["clorofila_media"].iloc[:i].mean())


def test_media_3anos_exclui_o_mes_corrente():
    m = mensal_sintetico()
    m["m36"] = media_movel_passada(m, "clorofila_media")
    p = m[m.latitude == -24.0].sort_values(["ano", "mes"]).reset_index(drop=True)
    assert p["m36"].iloc[:12].isna().all()  # menos de 12 meses anteriores
    for i in (12, 30, 50):
        assert p["m36"].iloc[i] == pytest.approx(p["clorofila_media"].iloc[max(0, i - 36):i].mean())


@pytest.mark.parametrize("ano_mudado", [2004, 2005])
def test_features_nao_enxergam_o_futuro(ano_mudado):
    """Mudar os dados de um ano não pode alterar nenhuma feature de anos anteriores."""
    m = mensal_sintetico()
    alterado = m.copy()
    futuro = alterado["ano"] >= ano_mudado
    for c in ("clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"):
        alterado.loc[futuro, c] = alterado.loc[futuro, c] * 10 + 7

    a = constroi_features(m)
    b = constroi_features(alterado)
    a, b = a[a.ano < ano_mudado].reset_index(drop=True), b[b.ano < ano_mudado].reset_index(drop=True)
    colunas = [c for c in a.columns if c != "floracao"]
    pd.testing.assert_frame_equal(a[colunas], b[colunas])


def test_features_do_mes_t_nao_dependem_da_clorofila_de_t():
    m = mensal_sintetico()
    alterado = m.copy()
    linha = (alterado.latitude == -24.0) & (alterado.ano == 2004) & (alterado.mes == 7)
    alterado.loc[linha, ["clorofila_media", "clorofila_maxima"]] = 99.0
    a = constroi_features(m).set_index(["latitude", "longitude", "ano", "mes"])
    b = constroi_features(alterado).set_index(["latitude", "longitude", "ano", "mes"])
    chave = (-24.0, -46.0, 2004, 7)
    for c in ("clorofila_media_historica_mes", "clorofila_media_3anos", "clorofila_media_ano_anterior"):
        assert a.loc[chave, c] == b.loc[chave, c]
    assert b.loc[chave, "floracao"] == 1  # o alvo sim muda


def test_chave_duplicada_e_recusada():
    m = mensal_sintetico()
    with pytest.raises(ValueError):
        constroi_features(pd.concat([m, m.iloc[:1]]))
