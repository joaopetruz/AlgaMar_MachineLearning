"""Testes com os dados reais (pulados se os arquivos não estiverem nesta máquina)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from algamar_ml.processamento.mensal import chave_ponto, processa

EQUIPE = Path(__file__).resolve().parents[3] / "cfd_navier_stokes" / "data" / "raw" / "ml_equipe"
PAINEL = Path(__file__).resolve().parents[1] / "dados" / "processados" / "painel_mensal.parquet"


@pytest.mark.skipif(not (EQUIPE / "clorofila_2024_sp.nc").exists(), reason="NetCDF da equipe ausentes")
def test_processamento_reproduz_a_tabela_mensal_da_equipe_em_2024():
    t = processa(xr.open_dataset(EQUIPE / "clorofila_2024_sp.nc"), xr.open_dataset(EQUIPE / "temperatura_2024_sp.nc"),
                 xr.open_dataset(EQUIPE / "salinidade_2024_sp.nc"))
    eq = pd.read_csv(EQUIPE / "dados_mensais_2000_2024_sp_com_salinidade.csv")
    eq = eq[eq.ano == 2024].copy()
    eq["latitude"], eq["longitude"] = chave_ponto(eq.latitude, eq.longitude)
    m = eq.merge(t, on=["latitude", "longitude", "ano", "mes"], suffixes=("_eq", "_novo"))
    assert len(m) == len(eq)
    for c in ("clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"):
        assert np.allclose(m[c + "_eq"], m[c + "_novo"], atol=1e-5), c


@pytest.mark.skipif(not PAINEL.exists(), reason="painel ainda não construído")
def test_painel_real_sem_chave_repetida_e_com_proveniencia():
    p = pd.read_parquet(PAINEL)
    assert not p.duplicated(["latitude", "longitude", "ano", "mes"]).any()
    assert p[["latitude", "longitude"]].drop_duplicates().shape[0] == 1790
    assert p["latitude"].between(-25.3, -23.3).all() and p["longitude"].between(-47.0, -44.8).all()
    assert (p.loc[p.ano <= 2024, "fonte_clorofila"].str.startswith("historico_equipe")).all()
    assert p["mes"].between(1, 12).all()
