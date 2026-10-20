import pandas as pd

tabela = pd.read_csv("dados/dados_tratados_2024.csv", parse_dates=["time"])

# Ver os 20 valores mais altos de clorofila
print("Top 20 valores mais altos de clorofila:")
print(tabela.sort_values("clorofila", ascending=False).head(20))

# Quantos valores estão acima de um limite "razoável" (ex: 30)
suspeitos = tabela[tabela["clorofila"] > 30]
print(f"\nQuantidade de valores acima de 30 mg/m³: {len(suspeitos)}")
print(f"Isso representa {len(suspeitos)/len(tabela)*100:.4f}% do total de dados")