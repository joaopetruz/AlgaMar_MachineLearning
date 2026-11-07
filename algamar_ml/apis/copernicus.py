"""Download das variáveis ambientais da Copernicus Marine (histórico e condição atual).

Para cada variável há um produto reprocessado (MY/REP, o mesmo da série histórica da
equipe) e um produto near-real-time (NRT). Para cada mês usa-se o MY/REP se ele cobrir o
mês inteiro; senão, o NRT. A escolha fica registrada na proveniência.

Login: o copernicusmarine usa as credenciais que o próprio usuário configurou
(`copernicusmarine login`). Este módulo nunca lê nem grava credenciais.

| variável | produto MY/REP (histórico)                                   | produto NRT (atual)                                          | resolução        | unidade |
|----------|--------------------------------------------------------------|--------------------------------------------------------------|------------------|---------|
| CHL      | cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D    | cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D   | 4 km, diária     | mg m-3  |
| SST      | METOFFICE-GLO-SST-L4-REP-OBS-SST                             | METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2                          | 0,05°, diária    | K       |
| SSS      | cmems_obs-mob_glo_phy-sss_my_multi_P1D                       | cmems_obs-mob_glo_phy-sss_nrt_multi_P1D                      | 0,125°, diária   | PSU     |
"""
import calendar
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Produto:
    dataset_id: str
    tipo: str          # "MY" (reprocessado) ou "NRT"
    variavel_origem: str


PRODUTOS = {
    "clorofila": [
        Produto("cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", "MY", "CHL"),
        Produto("cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "NRT", "CHL"),
    ],
    "temperatura": [
        Produto("METOFFICE-GLO-SST-L4-REP-OBS-SST", "MY", "analysed_sst"),
        Produto("METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2", "NRT", "analysed_sst"),
    ],
    "salinidade": [
        Produto("cmems_obs-mob_glo_phy-sss_my_multi_P1D", "MY", "sos"),
        Produto("cmems_obs-mob_glo_phy-sss_nrt_multi_P1D", "NRT", "sos"),
    ],
}


def _ms_para_data(x):
    return dt.datetime.fromtimestamp(x / 1000, dt.UTC).date() if x is not None else None


def cobertura(dataset_id):
    """(primeiro dia, último dia) disponíveis no catálogo da Copernicus (não precisa de login)."""
    import copernicusmarine as cm

    desc = cm.describe(dataset_id=dataset_id, disable_progress_bar=True)
    for produto in desc.products:
        for ds in produto.datasets:
            for versao in ds.versions:
                for parte in versao.parts:
                    for servico in parte.services:
                        for var in servico.variables:
                            for coord in var.coordinates:
                                if coord.coordinate_id == "time" and coord.minimum_value is not None:
                                    return _ms_para_data(coord.minimum_value), _ms_para_data(coord.maximum_value)
    raise RuntimeError(f"cobertura temporal não encontrada para {dataset_id}")


def limites_do_mes(ano, mes):
    return dt.date(ano, mes, 1), dt.date(ano, mes, calendar.monthrange(ano, mes)[1])


def escolhe_produto(variavel, ano, mes, coberturas):
    """MY/REP se cobrir o mês inteiro; senão NRT se cobrir; senão None (mês indisponível)."""
    inicio, fim = limites_do_mes(ano, mes)
    for produto in PRODUTOS[variavel]:
        primeiro, ultimo = coberturas[produto.dataset_id]
        if primeiro <= inicio and ultimo >= fim:
            return produto
    return None


def caixa(cfg, variavel):
    area = cfg["area"]
    margem = 0.0 if variavel == "clorofila" else area["margem_graus"]
    return dict(
        minimum_longitude=area["lon_min"] - margem, maximum_longitude=area["lon_max"] + margem,
        minimum_latitude=area["lat_min"] - margem, maximum_latitude=area["lat_max"] + margem,
    )


def baixa_mes(cfg, variavel, produto, ano, mes):
    """Baixa um mês de dados diários. Pula se o arquivo já existe. Devolve o caminho."""
    import copernicusmarine as cm

    destino = Path(cfg["caminhos"]["brutos"]) / variavel / produto.dataset_id
    destino.mkdir(parents=True, exist_ok=True)
    nome = f"{variavel}_{ano}-{mes:02d}.nc"
    if (destino / nome).exists():
        return destino / nome
    inicio, fim = limites_do_mes(ano, mes)
    extra = {"minimum_depth": 0.0, "maximum_depth": 1.0} if variavel in ("salinidade", "ph") else {}
    cm.subset(
        dataset_id=produto.dataset_id,
        variables=[produto.variavel_origem],
        start_datetime=f"{inicio}T00:00:00",
        end_datetime=f"{fim}T00:00:00",
        output_directory=str(destino),
        output_filename=nome,
        disable_progress_bar=True,
        **caixa(cfg, variavel),
        **extra,
    )
    registro = {"dataset_id": produto.dataset_id, "tipo": produto.tipo, "variavel": produto.variavel_origem,
                "ano": ano, "mes": mes, "baixado_em": dt.datetime.now(dt.UTC).isoformat()}
    (destino / nome.replace(".nc", ".json")).write_text(json.dumps(registro, indent=1), encoding="utf-8")
    return destino / nome
