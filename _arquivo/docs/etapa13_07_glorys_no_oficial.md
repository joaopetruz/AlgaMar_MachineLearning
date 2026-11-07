# Etapa 13.7 — física do GLORYS no modelo oficial (baseline-hist-api-v1)

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage`. Nada foi enviado ao GitHub.

**O que aconteceu antes:** a produção CROCO 2014–2024, de ~52 dias de máquina, foi descartada pelo Tulio. A regra agora é que nada passe de 1–2 h. Por isso a física vem da reanálise GLORYS12 mensal, já baixada.

**Código:**
- `algamar_ml/glorys_ml.py` e as flags `use_glorys_ts` / `use_glorys_mld` em `config/ml.toml`. Ficam desligadas por padrão, e o baseline continua idêntico.
- Script: `etapa13_07_glorys_oficial.py`. Leva 10 min.
- Saídas: `resultados/etapa13_07/`.

**Desenho:**
- O modelo oficial foi treinado com o mesmo config em três variantes, todas com as mesmas linhas:
  - OF0: o oficial;
  - OF1: OF0 + MLD do GLORYS em t0;
  - OF2: OF0 + T/S + MLD do GLORYS em t0.
- Teste: emissões de 2022-01 a 2024-12, 773 280 linhas. O GLORYS mensal termina em 2024-12.
- IC 95 % por bootstrap em blocos (ano × região × trimestre).

## Resultado (MEDIDO)

| modelo | PR-AUC | ROC-AUC | Brier | F1 | Δ PR-AUC vs OF0 [IC 95 %] |
|---|---|---|---|---|---|
| OF0 oficial | 0,7810 | 0,9058 | 0,0895 | 0,657 | — |
| OF1 + MLD GLORYS | 0,7840 | 0,9095 | 0,0891 | 0,659 | **+0,003 [+0,0005; +0,006]** (99,7 % > 0) |
| OF2 + T/S + MLD GLORYS | 0,7809 | 0,9040 | 0,0889 | 0,670 | −0,000 [−0,004; +0,004] |

- **Por horizonte:** a MLD dá de +0,002 a +0,004 em todos os h = 1…12. T/S oscila em torno de zero.
- **Comparação com a 13.6:** o resultado confirma o que ela mostrou. O modelo oficial já usa SST de satélite e anomalias em t0, então o T/S do GLORYS não acrescenta nada. Só a MLD traz informação nova, e o ganho é consistente, mas muito pequeno (+0,4 %).

## Recomendação

**Manter o oficial sem física (OF0).** O ganho de +0,003 não paga a dependência operacional:
- o GLORYS MY sai com 1–2 meses de atraso, então não há MLD para o mês de emissão mais recente;
- seria preciso trocar pelo produto de análise/previsão (ANFC), que é outra fonte e exige revalidação.

As flags ficam no código para religar se surgir uma MLD operacional.
