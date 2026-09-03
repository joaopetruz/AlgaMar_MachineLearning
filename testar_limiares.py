import pandas as pd

tabela = pd.read_csv("dados/dados_features_2024.csv")

print("Quantidade de casos por limiar de clorofila:\n")
for limiar in [2, 3, 5, 8, 10, 15, 20]:
    casos = (tabela["clorofila"] > limiar).sum()
    porcentagem = casos / len(tabela) * 100
    print(f"Limiar {limiar:>3} mg/m³: {casos:>6} casos ({porcentagem:.3f}%)")