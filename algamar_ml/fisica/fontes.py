"""Adaptadores de fontes físicas. Cada um entrega uma tabela padrão:
    latitude, longitude, ano, mes, <features no nome do contrato>   (mensal)
ou  time (UTC), latitude, longitude, <features>                       (diária/instantânea)

- FonteTabelaContrato13: Parquet/CSV exportado pelo cfd_navier_stokes no contrato 13.0
  (CROCO D2), diário ou mensal; colunas mensais `<coluna>_mean` são aceitas.
- FonteGradeUV: NetCDF em grade regular lat/lon com u e v (ex.: solver Navier–Stokes 2D
  ou outro modelo); calcula velocidade, direção, energia cinética, vorticidade,
  divergência e Okubo-Weiss na grade nativa e devolve em formato longo.
Para um solver novo, basta outro adaptador com o mesmo método `carrega()`.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml.fisica import contrato as ctr
from algamar_ml.fisica.derivadas import deriva


class FonteFisica(ABC):
    nome: str
    versao: str

    @abstractmethod
    def carrega(self) -> pd.DataFrame:
        ...

    def descricao(self):
        return {"fonte": self.nome, "versao": self.versao, "classe": type(self).__name__}


class FonteTabelaContrato13(FonteFisica):
    def __init__(self, caminho, nome="croco_d2", versao="contrato-13.0"):
        self.caminho, self.nome, self.versao = Path(caminho), nome, versao

    def carrega(self):
        t = pd.read_parquet(self.caminho) if self.caminho.suffix == ".parquet" else pd.read_csv(self.caminho)
        sufixo = {c: c[: -len("_mean")] for c in t.columns if c.endswith("_mean")}
        t = t.rename(columns=sufixo)
        if "time" in t.columns and "ano" not in t.columns:
            t["time"] = pd.to_datetime(t["time"], utc=True)
        t = ctr.padroniza(t)
        # derivadas que o export não traz, a partir das colunas presentes
        if {"u", "v"}.issubset(t.columns):
            vel = np.hypot(t["u"], t["v"])
            if "velocidade" not in t:
                t["velocidade"] = vel
            if "direcao_sin" not in t:
                ang = np.arctan2(t["u"], t["v"])
                t["direcao_sin"] = np.where(vel > 0, np.sin(ang), np.nan)
                t["direcao_cos"] = np.where(vel > 0, np.cos(ang), np.nan)
            if "energia_cinetica" not in t:
                t["energia_cinetica"] = 0.5 * (t["u"] ** 2 + t["v"] ** 2)
        return t


class FonteGradeUV(FonteFisica):
    def __init__(self, caminho, nome="navier_stokes", versao="desconhecida", var_u="u", var_v="v", var_t=None):
        self.caminho, self.nome, self.versao = Path(caminho), nome, versao
        self.var_u, self.var_v, self.var_t = var_u, var_v, var_t

    def carrega(self):
        import xarray as xr

        ds = xr.open_dataset(self.caminho)
        u, v = ds[self.var_u].transpose("time", "latitude", "longitude"), ds[self.var_v].transpose("time", "latitude", "longitude")
        lat, lon = ds["latitude"].to_numpy(), ds["longitude"].to_numpy()
        campos = {"u": u.to_numpy(), "v": v.to_numpy()}
        temp = ds[self.var_t].transpose("time", "latitude", "longitude").to_numpy() if self.var_t else None
        campos.update(deriva(campos["u"], campos["v"], lat, lon, temperatura=temp))
        if temp is not None:
            campos["temperature_surface"] = temp
        tempo = pd.to_datetime(ds["time"].to_numpy(), utc=True)
        nt, ny, nx = campos["u"].shape
        saida = pd.DataFrame({
            "time": np.repeat(tempo, ny * nx),
            "latitude": np.tile(np.repeat(lat, nx), nt),
            "longitude": np.tile(np.tile(lon, ny), nt),
        })
        for nome, a in campos.items():
            saida[nome] = a.reshape(-1)
        return saida
