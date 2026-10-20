"""Baixa da Copernicus os meses posteriores ao histórico da equipe (condição ambiental atual).

Uso:
  python atualizar_dados_atuais.py                       # de 2025-01 até o último mês completo
  python atualizar_dados_atuais.py --inicio 2024-01 --fim 2024-12 --tipo NRT   # sobreposição MY×NRT
"""
import argparse
import datetime as dt
import json

from algamar_ml import config
from algamar_ml.apis.copernicus import PRODUTOS, baixa_mes, cobertura, escolhe_produto


def meses(inicio, fim):
    a, m = inicio
    while (a, m) <= fim:
        yield a, m
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)


def ano_mes(texto):
    a, m = texto.split("-")
    return int(a), int(m)


hoje = dt.date.today()
ultimo_completo = (hoje.year, hoje.month - 1) if hoje.month > 1 else (hoje.year - 1, 12)

ap = argparse.ArgumentParser()
ap.add_argument("--inicio", default="2025-01")
ap.add_argument("--fim", default=f"{ultimo_completo[0]}-{ultimo_completo[1]:02d}")
ap.add_argument("--tipo", choices=["auto", "MY", "NRT"], default="auto",
                help="auto = MY/REP quando cobre o mês, senão NRT")
ap.add_argument("--variaveis", default="clorofila,temperatura,salinidade")
args = ap.parse_args()

cfg = config.carrega()
coberturas = {p.dataset_id: cobertura(p.dataset_id) for ps in PRODUTOS.values() for p in ps}
print(json.dumps({k: [str(a), str(b)] for k, (a, b) in coberturas.items()}, indent=1))

faltando = []
for variavel in args.variaveis.split(","):
    for ano, mes in meses(ano_mes(args.inicio), ano_mes(args.fim)):
        if args.tipo == "auto":
            produto = escolhe_produto(variavel, ano, mes, coberturas)
        else:
            produto = next(p for p in PRODUTOS[variavel] if p.tipo == args.tipo)
        if produto is None:
            faltando.append(f"{variavel} {ano}-{mes:02d}")
            continue
        caminho = baixa_mes(cfg, variavel, produto, ano, mes)
        print(f"{variavel} {ano}-{mes:02d} {produto.tipo:3s} {caminho.name}")

if faltando:
    print("\nSem produto que cubra o mês inteiro (não baixado):", ", ".join(faltando))
