"""Registro de modelos versionado e imutável (Etapa 13.8).

Cada versão vive em <modelos>/<versao>/ com modelo.json (XGBoost), cartao.json e MANIFESTO.json (SHA256 dos dois).
14.1: um ensemble de ordens tem modelo.json = descritor e membro_XX.json; o manifesto cobre também os membros.
- registra(): recusa uma versão que já existe. Um modelo antigo nunca é sobrescrito.
- carrega(): confere os SHA256 do manifesto antes de usar. Um arquivo alterado ou corrompido gera ErroRegistro.
- manifesta(): cria o manifesto de uma versão antiga que ainda não tem um, sem tocar nos arquivos do modelo.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

from .modelos import treino as tr

ARQUIVOS = ("modelo.json", "cartao.json")


class ErroRegistro(Exception):
    pass


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _manifesto(pasta, origem, arquivos=ARQUIVOS):
    m = {"versao": pasta.name, "arquivos": {a: sha256(pasta / a) for a in arquivos},
         "criado_em": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "origem": origem}
    (pasta / "MANIFESTO.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m


def registra(modelo, cartao, raiz, versao):
    pasta = Path(raiz) / versao
    if pasta.exists():
        raise ErroRegistro(f"a versão '{versao}' já existe em {pasta}; modelos antigos não são sobrescritos — use outra versão")
    tr.salva(modelo, {**cartao, "versao": versao}, pasta)
    return _manifesto(pasta, "registra()", ARQUIVOS + tuple(getattr(modelo, "arquivos_membros", ())))


def manifesta(raiz, versao):
    pasta = Path(raiz) / versao
    if (pasta / "MANIFESTO.json").exists():
        return json.loads((pasta / "MANIFESTO.json").read_text(encoding="utf-8"))
    if not all((pasta / a).exists() for a in ARQUIVOS):
        raise ErroRegistro(f"versão '{versao}' incompleta em {pasta}")
    return _manifesto(pasta, "manifesta() de versão pré-existente (arquivos não alterados)")


def carrega(raiz, versao):
    pasta = Path(raiz) / versao
    if not pasta.exists():
        raise ErroRegistro(f"versão '{versao}' não encontrada em {raiz}")
    man = manifesta(raiz, versao)
    for a, h in man["arquivos"].items():
        if not (pasta / a).exists() or sha256(pasta / a) != h:
            raise ErroRegistro(f"'{versao}/{a}' difere do MANIFESTO (arquivo alterado ou corrompido)")
    try:
        modelo, cartao = tr.carrega(pasta)
    except Exception as e:
        raise ErroRegistro(f"modelo '{versao}' ilegível: {type(e).__name__}: {e}") from e
    fora = set(getattr(modelo, "arquivos_membros", ())) - set(man["arquivos"])
    if fora:
        raise ErroRegistro(f"'{versao}': membros fora do MANIFESTO: {sorted(fora)}")
    return modelo, cartao, man


def lista(raiz):
    out = []
    for p in sorted(Path(raiz).glob("*/cartao.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        out.append({"versao": p.parent.name, "status": c.get("status"), "salvo_em": c.get("salvo_em"),
                    "manifesto": (p.parent / "MANIFESTO.json").exists()})
    return out
