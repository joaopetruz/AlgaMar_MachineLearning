"""Agrega por mês os dados da estação meteorológica do CEBIMar/USP (Praia do Segredo, São Sebastião, 23°49'25" S,
45°25'18" W; Davis Vantage Pro2, a cada 10 min, desde 29/05/2013).

Os dados ficam SÓ no computador (dados/cebimar/ está no .gitignore): a fonte não permite reproduzir os dados brutos.
Citação: "Dados coletados, gerenciados e cedidos pela equipe técnica do Centro de Biologia Marinha da Universidade de
São Paulo". Planilhas: https://cebimar.usp.br/pesquisa-monitoramento-ambiental-estacao-meteorologica-do-cebimarusp/

Uso: python processar_cebimar.py  → dados/cebimar/cebimar_mensal.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml import config

PASTA = Path(config.carrega()["caminhos"]["painel"]).parent.parent / "cebimar"
# posição das colunas (a ordem do Davis é a mesma em todos os anos; os nomes mudam, em 2016 estão em inglês)
COLUNAS = {2: "temp_ar", 7: "vento_kmh", 8: "vento_dir", 16: "pressao", 17: "chuva_mm", 19: "radiacao"}
SUL = {"S", "SSW", "SW", "SSE", "SE", "SSO", "SO"}   # vento vindo do quadrante sul (frente fria)


def le_ano(arq):
    abas = pd.read_excel(arq, sheet_name=None, header=None)
    partes = []
    for df in abas.values():
        if df.shape[1] < 20:
            continue
        d = df[list(COLUNAS)].rename(columns=COLUNAS)
        dia = pd.to_datetime(df[0], errors="coerce")   # linhas de cabeçalho viram NaT e saem
        d["t"] = dia + pd.to_timedelta(df[1].astype(str), errors="coerce")
        partes.append(d)
    d = pd.concat(partes, ignore_index=True).dropna(subset=["t"])
    for c in ("temp_ar", "vento_kmh", "pressao", "chuva_mm", "radiacao"):
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "."), errors="coerce")
    return d


def mensal(d):
    d = d.set_index("t").sort_index()
    d = d[~d.index.duplicated()]
    dia = pd.DataFrame({"chuva": d.chuva_mm.resample("D").sum(min_count=1),
                        "n": d.temp_ar.resample("D").count(),
                        "sul": d.vento_dir.isin(SUL).resample("D").mean(),
                        "temp": d.temp_ar.resample("D").mean(),
                        "rad": d.radiacao.resample("D").mean()})
    dia = dia[dia.n >= 100]   # dia com pelo menos ~70% das leituras de 10 min
    m = pd.DataFrame({
        "dias_com_dado": dia.n.resample("MS").count(),
        "chuva_mm": dia.chuva.resample("MS").sum(),
        "chuva_max_dia_mm": dia.chuva.resample("MS").max(),
        "dias_chuva_forte": (dia.chuva >= 20).resample("MS").sum(),
        "frac_vento_sul": dia.sul.resample("MS").mean(),
        "temp_ar_c": dia.temp.resample("MS").mean(),
        "radiacao_w_m2": dia.rad.resample("MS").mean(),
    })
    # mês com poucos dias medidos: a chuva total fica proporcional aos dias medidos
    m["chuva_mm"] = m.chuva_mm * m.index.days_in_month / m.dias_com_dado.replace(0, np.nan)
    return m[m.dias_com_dado >= 20]


if __name__ == "__main__":
    arqs = sorted(PASTA.glob("Dados-Meteorologicos-Cebimar-*.xlsx"))
    m = pd.concat([mensal(le_ano(a)) for a in arqs]).sort_index()
    m = m[~m.index.duplicated()]
    clim = m.groupby(m.index.month).chuva_mm.median()
    m["chuva_anom_pct"] = 100 * (m.chuva_mm / m.index.month.map(clim).values - 1)
    m.to_parquet(PASTA / "cebimar_mensal.parquet")
    pd.set_option("display.width", 200)
    print(m.round(1).to_string())
