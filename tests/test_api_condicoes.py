import json

import pytest
from fastapi import HTTPException

from algamar_ml import api_condicoes


def municipio(nome, macro, especie):
    return {"municipio": nome, "macro_region": macro, "latitude": -24.0, "longitude": -46.3, "especie": especie,
            "agora": {"mes": "2026-10", "indice": 0.5, "nivel": "parcialmente propício"}, "proximos": []}


@pytest.fixture()
def saida(tmp_path, monkeypatch):
    arq = tmp_path / "condicoes_propicias.json"
    arq.write_text(json.dumps({"gerado_em": "x", "aviso": "a", "cortes": {"propicio": 0.7}, "municipios": [
        municipio("Santos", "Baixada Santista", "Dinophysis"), municipio("Santos", "Baixada Santista", "Pseudo-nitzschia"),
        municipio("Ubatuba", "Litoral Norte", "Dinophysis")]}), encoding="utf-8")
    monkeypatch.setattr(api_condicoes, "arquivo", lambda: arq)
    return arq


def test_filtra_por_especie_e_regiao(saida):
    r = api_condicoes.monta("dinophysis", None)
    assert [m["municipio"] for m in r["data"]] == ["Santos", "Ubatuba"]
    assert r["especies"] == ["Dinophysis", "Pseudo-nitzschia"]
    assert [m["especie"] for m in api_condicoes.monta(None, "Baixada Santista")["data"]] == ["Dinophysis", "Pseudo-nitzschia"]
    assert len(api_condicoes.monta(None, "ubatuba")["data"]) == 1


def test_sem_arquivo_avisa(tmp_path, monkeypatch):
    monkeypatch.setattr(api_condicoes, "arquivo", lambda: tmp_path / "nao_existe.json")
    with pytest.raises(HTTPException) as e:
        api_condicoes.monta()
    assert e.value.status_code == 503
