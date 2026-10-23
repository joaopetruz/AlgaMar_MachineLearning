import numpy as np
import pytest
from fastapi import HTTPException

from algamar_ml import api_teste, config

cfg = config.carrega()
TEM_DADOS = cfg["caminhos"]["painel"].exists() and (cfg["caminhos"]["modelos"] / cfg["modelo"]["oficial"]).exists()


def test_mes_invalido_da_422():
    for texto in ("2025-13", "2025", "abc"):
        with pytest.raises(HTTPException) as e:
            api_teste._mes(texto)
        assert e.value.status_code == 422


@pytest.mark.skipif(not TEM_DADOS, reason="painel/modelo oficial ficam fora do git")
def test_hindcast_nao_ve_o_futuro(monkeypatch):
    """Trocar tudo depois de t0 por lixo não pode mudar a previsão feita em t0."""
    antes = api_teste.hindcast("2025-03")
    caminho = cfg["caminhos"]["painel"]
    painel = api_teste._painel(str(caminho), caminho.stat().st_mtime).copy()
    futuro = painel["_t"] > 2025 * 12 + 2
    rng = np.random.default_rng(0)
    for c in ["clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"]:
        painel.loc[futuro, c] = rng.uniform(0, 50, futuro.sum())
    monkeypatch.setattr(api_teste, "_painel", lambda *a: painel)
    depois = api_teste.hindcast("2025-03")
    prob = lambda r: [x["prob_prevista"] for x in r["regioes"]["Litoral todo"]]
    assert prob(antes) == prob(depois)
    assert antes["as_cegas"] and len(antes["regioes"]["Litoral todo"]) == 12


@pytest.mark.skipif(not TEM_DADOS, reason="painel/modelo oficial ficam fora do git")
def test_emissao_vista_no_treino_e_avisada():
    r = api_teste.hindcast("2015-06")
    assert not r["as_cegas"] and r["meses_ja_vistos_no_treino"] == 12
