import pandas as pd

from algamar_ml.features import constroi_features

# Features sem vazamento: cada linha usa só meses anteriores a ela (ver algamar_ml/features.py)
mensal = pd.read_csv("dados/dados_mensais_2000_2024_sp_com_salinidade.csv")
print(f"Linhas mensais: {len(mensal)}")

tabela = constroi_features(mensal)
print(f"Linhas com features (após remover quem não tem o ano anterior): {len(tabela)}")

print(f"\nDistribuição do target:")
print(tabela["floracao"].value_counts())
print(f"Porcentagem de floração: {tabela['floracao'].mean()*100:.2f}%")

tabela.to_csv("dados/dados_features_sazonal_sp_com_salinidade.csv", index=False)
print("\nArquivo salvo: dados/dados_features_sazonal_sp_com_salinidade.csv")
