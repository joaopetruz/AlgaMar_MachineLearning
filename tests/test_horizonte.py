import numpy as np
import pandas as pd
import pytest

from algamar_ml.features.horizonte import (
    climatologia_passada, constroi_linhas, cubo_do_painel, indice_mes, media_movel, nomes_features, texto_mes,
)

LIMIAR = 1.5


def painel_sintetico(anos=range(2000, 2006), pontos=((-24.0, -46.0), (-24.5, -45.5), (-23.8, -45.2)), seed=1):
    rng = np.random.default_rng(seed)
    linhas = []
    for lat, lon in pontos:
        for ano in anos:
            for mes in range(1, 13):
                chl = float(rng.gamma(2.0, 0.6))
                linhas.append(dict(latitude=lat, longitude=lon, ano=ano, mes=mes, clorofila_media=chl,
                                   clorofila_maxima=chl * 1.7, temperatura_media=22 + rng.normal(),
                                   salinidade_media=35 + rng.normal(0, 0.3)))
    return pd.DataFrame(linhas)


def test_indices_de_mes_e_virada_de_ano():
    t = indice_mes(2025, 11)
    assert str(texto_mes(t + 2)) == "2026-01"
    assert texto_mes([t + 13])[0] == "2026-12"


def test_alvo_e_t0_mais_h_inclusive_na_virada_de_ano():
    p = painel_sintetico()
    cubo = cubo_do_painel(p)
    t0 = indice_mes(2003, 11)
    linhas = constroi_linhas(cubo, LIMIAR, [t0], horizontes=[1, 2, 3])
    for h in (1, 2, 3):
        linha = linhas[(linhas.h == h) & (linhas.latitude == np.float32(-24.0))].iloc[0]
        ano, mes = divmod(t0 + h, 12)
        esperado = p[(p.latitude == -24.0) & (p.ano == ano) & (p.mes == mes + 1)]["clorofila_maxima"].iloc[0] > LIMIAR
        assert linha["floracao"] == float(esperado)
        assert linha["mes_alvo"] == mes + 1
        assert linha["tau"] == t0 + h


def test_nenhuma_feature_usa_dado_depois_de_t0():
    """Mudar tudo depois de t0* não pode mudar nenhuma feature de emissões t0 <= t0*."""
    p = painel_sintetico()
    t_corte = indice_mes(2003, 6)
    alterado = p.copy()
    futuro = indice_mes(alterado.ano, alterado.mes) > t_corte
    for c in ("clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"):
        alterado.loc[futuro, c] = alterado.loc[futuro, c] * 7 + 3
    emissoes = range(indice_mes(2001, 1), t_corte + 1)
    a = constroi_linhas(cubo_do_painel(p), LIMIAR, emissoes)
    b = constroi_linhas(cubo_do_painel(alterado), LIMIAR, emissoes)
    feats = [c for c in nomes_features({}) if c in a.columns]
    pd.testing.assert_frame_equal(a[feats], b[feats])
    assert not a["floracao"].equals(b["floracao"])  # o alvo (futuro) sim muda


def test_climatologia_so_anos_anteriores_e_media_movel_inclui_t0():
    a = np.arange(36, dtype=np.float32).reshape(36, 1)
    c = climatologia_passada(a)
    assert np.isnan(c[5, 0]) and c[17, 0] == 5 and c[29, 0] == pytest.approx((5 + 17) / 2)
    m = media_movel(a, 3, 2)
    assert np.isnan(m[0, 0]) and m[1, 0] == 0.5 and m[10, 0] == pytest.approx(9.0)


def test_previsao_alem_do_painel_tem_features_e_alvo_vazio():
    p = painel_sintetico()
    cubo = cubo_do_painel(p)
    linhas = constroi_linhas(cubo, LIMIAR, [cubo.t_fim])
    assert set(linhas.h) == set(range(1, 13)) and linhas["floracao"].isna().all()
    for c in ("clim_chl_media_alvo", "freq_floracao_alvo", "chl_max_alvo_ano_anterior"):
        assert linhas[c].notna().all()
    # mesmo mês do ano anterior = mês t0+h-12 do painel
    l12 = linhas[(linhas.h == 12) & (linhas.latitude == np.float32(-24.0))].iloc[0]
    assert l12["chl_max_alvo_ano_anterior"] == pytest.approx(
        p[(p.latitude == -24.0) & (p.ano == 2005) & (p.mes == 12)]["clorofila_maxima"].iloc[0], rel=1e-6)


def test_previsao_alem_do_painel_igual_ao_calculo_dentro_do_painel():
    """A climatologia do mês-alvo calculada além do painel bate com a calculada dentro dele."""
    p = painel_sintetico()
    curto = p[indice_mes(p.ano, p.mes) <= indice_mes(2004, 12)]
    t0 = indice_mes(2004, 12)
    fora = constroi_linhas(cubo_do_painel(curto), LIMIAR, [t0]).sort_values(["h", "point_id"]).reset_index(drop=True)
    dentro = constroi_linhas(cubo_do_painel(p), LIMIAR, [t0]).sort_values(["h", "point_id"]).reset_index(drop=True)
    for c in ("clim_chl_media_alvo", "clim_chl_max_alvo", "freq_floracao_alvo", "clim_sst_alvo", "floracao_alvo_ano_anterior"):
        np.testing.assert_allclose(fora[c], dentro[c], rtol=1e-5)


def test_horizonte_acima_de_12_e_recusado():
    cubo = cubo_do_painel(painel_sintetico())
    with pytest.raises(ValueError):
        constroi_linhas(cubo, LIMIAR, [cubo.t_fim], horizontes=[13])


def test_mes_faltando_vira_nan_e_nao_outro_mes():
    p = painel_sintetico()
    p = p[~((p.latitude == -24.0) & (p.ano == 2003) & (p.mes == 4))]
    cubo = cubo_do_painel(p)
    linhas = constroi_linhas(cubo, LIMIAR, [indice_mes(2003, 5)], horizontes=[1])
    l = linhas[linhas.latitude == np.float32(-24.0)].iloc[0]
    assert np.isnan(l["chl_media_t0m1"])


def test_flags_controlam_os_grupos():
    so_base = nomes_features({"use_api_features": False, "use_historical_features": False})
    assert so_base == ["h", "mes_emissao", "mes_alvo", "latitude", "longitude"]
    com_fisica = nomes_features({"use_navier_stokes_features": True}, fisicas=["ns_speed_t0"])
    assert "ns_speed_t0" in com_fisica and "sst_t0" in com_fisica


def test_toda_feature_do_modelo_esta_no_contrato():
    from algamar_ml.features.contrato_ml import tabela
    nomes = set(tabela()["nome"])
    for f in nomes_features({"use_api_features": True, "use_historical_features": True}):
        assert f in nomes, f


def test_flag_de_salinidade_remove_features_sss():
    nomes = nomes_features({"use_salinity": False})
    assert not [n for n in nomes if n.startswith("sss")] and "sst_t0" in nomes


def test_e0_forecast_h1_igual_ao_e0_corrigido():
    """13.3: em h = 1 (emissão T-1), as features do E0_forecast são exatamente as do E0_corrigido da linha T."""
    from algamar_ml.features.horizonte import E0_FORECAST_FEATURES
    from algamar_ml.features.sazonal import constroi_features
    p = painel_sintetico()
    hs = [(-24.0, -46.0), (-23.8, -45.2)]
    e0 = constroi_features(p, hotspots=hs).set_index(["latitude", "longitude", "ano", "mes"])
    cubo = cubo_do_painel(p)
    T = indice_mes(2004, 8)
    lin = constroi_linhas(cubo, LIMIAR, [T - 1], horizontes=[1], hotspots=hs)
    mapa = {"chl_media_alvo_ano_anterior": "clorofila_media_ano_anterior", "chl_max_alvo_ano_anterior": "clorofila_maxima_ano_anterior",
            "sst_alvo_ano_anterior": "temperatura_ano_anterior", "sss_alvo_ano_anterior": "salinidade_ano_anterior",
            "clim_chl_media_alvo": "clorofila_media_historica_mes", "clim_sss_alvo": "salinidade_media_historica_mes",
            "chl_media_36m": "clorofila_media_3anos", "sss_36m": "salinidade_media_3anos", "dist_hotspot": "dist_hotspot"}
    for r in lin.itertuples():
        ref = e0.loc[(round(float(r.latitude), 1), round(float(r.longitude), 1), 2004, 8)]
        for novo, antigo in mapa.items():
            assert getattr(r, novo) == pytest.approx(ref[antigo], rel=1e-5, abs=1e-5), novo
        assert r.mes_alvo == 8 and r.h == 1
    assert set(mapa) | {"h", "mes_alvo", "latitude", "longitude"} == set(E0_FORECAST_FEATURES)


def test_e0_forecast_sem_dado_depois_de_t0_em_todos_os_horizontes():
    """Para cada h = 1..12: mudar qualquer mês depois de t0 não muda as features E0 da emissão t0."""
    from algamar_ml.features.horizonte import E0_FORECAST_FEATURES
    p = painel_sintetico()
    t0 = indice_mes(2004, 3)
    alt = p.copy()
    fut = indice_mes(alt.ano, alt.mes) > t0
    for c in ("clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"):
        alt.loc[fut, c] = alt.loc[fut, c] * 7 + 3
    hs = [(-24.0, -46.0)]
    a = constroi_linhas(cubo_do_painel(p), LIMIAR, [t0], hotspots=hs)
    b = constroi_linhas(cubo_do_painel(alt), LIMIAR, [t0], hotspots=hs)
    for h in range(1, 13):
        pd.testing.assert_frame_equal(a[a.h == h][E0_FORECAST_FEATURES].reset_index(drop=True),
                                      b[b.h == h][E0_FORECAST_FEATURES].reset_index(drop=True))


def test_versao_enxuta_tira_so_as_redundantes():
    from algamar_ml.features.horizonte import REDUNDANTES
    base = {"use_historical_features": True, "use_api_features": True, "use_salinity": False}
    cheia = nomes_features(base)
    enxuta = nomes_features({**base, "use_features_redundantes": False})
    assert len(cheia) == 25 and len(enxuta) == 19
    assert set(cheia) - set(enxuta) == set(REDUNDANTES) - {"sss_t0m1"}


def test_grupos_sem_sinal_sao_recusados():
    import pytest
    from algamar_ml import config, pipeline
    cfg = config.carrega()
    assert pipeline.flags_de(cfg)["use_croco_current"] is False
    for flag in cfg["features"]["sem_sinal"]:
        c = {**cfg, "features": {**cfg["features"], flag: True}}
        with pytest.raises(pipeline.FeatureSemSinal):
            pipeline.flags_de(c)
        c["features"]["permitir_sem_sinal"] = True
        assert pipeline.flags_de(c)[flag] is True


def test_area_sp_total_inclui_litoral_sul_e_exclui_parana():
    from algamar_ml import config
    from algamar_ml.espacial import dentro_sp
    a = config.carrega()["area"]
    lat = [-24.0, -24.9, -25.15, -25.45, -25.0, -23.2]     # Santos, Ilha Comprida, Cananéia, PR, oceano a −46, fora (norte)
    lon = [-46.3, -47.6, -47.95, -48.15, -46.0, -45.0]
    assert dentro_sp(lat, lon, a).tolist() == [True, True, True, False, True, False]
