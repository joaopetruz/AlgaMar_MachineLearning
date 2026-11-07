"""Camada intermediária entre o solver físico (Navier–Stokes / CROCO) e o ML.

    solver → fontes.py (adaptador lê a saída) → derivadas.py (u,v → velocidade, vorticidade, …)
           → contrato.py (padroniza nomes/unidades e valida) → alinhamento.py (mês civil + pontos do ML)
           → feature_store.py (grava Parquet + manifesto de proveniência) → ML lê a store

O ML nunca lê a saída do solver diretamente: trocar o solver = escrever outro adaptador.
"""
