import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados já tratados
tabela = pd.read_csv("dados/dados_tratados_2024.csv", parse_dates=["time"])

print("Resumo geral dos dados:")
print(tabela.describe())

# ===== 1. Média diária de clorofila e temperatura (toda a região) =====
media_diaria = tabela.groupby("time")[["clorofila", "temperatura_celsius"]].mean()

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(media_diaria.index, media_diaria["clorofila"], color="green")
axs[0].set_title("Clorofila-a média diária — 2024 (região SP/RJ)")
axs[0].set_ylabel("Clorofila (mg/m³)")

axs[1].plot(media_diaria.index, media_diaria["temperatura_celsius"], color="orange")
axs[1].set_title("Temperatura da água média diária — 2024")
axs[1].set_ylabel("Temperatura (°C)")
axs[1].set_xlabel("Data")

plt.tight_layout()
plt.savefig("sazonalidade_2024.png", dpi=150)
print("\nGráfico salvo: sazonalidade_2024.png")
plt.show()

# ===== 2. Relação entre temperatura e clorofila =====
plt.figure(figsize=(8, 6))
amostra = tabela.sample(5000, random_state=42)  # amostra para não pesar o gráfico
plt.scatter(amostra["temperatura_celsius"], amostra["clorofila"], alpha=0.3, s=5, color="teal")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Clorofila (mg/m³)")
plt.title("Relação entre Temperatura e Clorofila-a")
plt.savefig("correlacao_temp_clorofila.png", dpi=150)
print("Gráfico salvo: correlacao_temp_clorofila.png")
plt.show()

# ===== 3. Correlação numérica =====
correlacao = tabela[["clorofila", "temperatura_celsius"]].corr()
print("\nCorrelação entre as variáveis:")
print(correlacao)

# ===== 4. Identificar os dias com maior concentração de clorofila (possível floração) =====
top_dias = media_diaria.sort_values("clorofila", ascending=False).head(10)
print("\nTop 10 dias com maior clorofila média (possíveis floramentos):")
print(top_dias)