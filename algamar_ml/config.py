"""Leitura da configuração (config/ml.toml) com caminhos resolvidos a partir da raiz do repositório."""
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PADRAO = RAIZ / "config" / "ml.toml"


def carrega(caminho=None):
    with open(caminho or PADRAO, "rb") as f:
        cfg = tomllib.load(f)
    cfg["caminhos"] = {k: (RAIZ / v) for k, v in cfg["caminhos"].items()}
    return cfg
