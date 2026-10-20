# Etapa 13.6 — Experimentos E0–E7: a física do oceano melhora o ML?

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage`. Nada foi enviado ao GitHub.

**Scripts:**
- `etapa13_06_experimentos.py --fonte glorys|croco`: o mesmo roteiro para as duas fontes;
- `etapa13_06_horizonte.py`;
- `baixar_glorys_mensal.py`.

**Saídas** em `resultados/etapa13_06/glorys/`:
- `tabela_experimentos.csv` e `metricas_por_seed.csv`;
- `comparacoes_pareadas_extras.csv`;
- `estratificacoes.csv` e `por_horizonte.csv`;
- `resultados.json`.

---

## 0. O que pôde e o que não pôde ser executado

- **E1–E6 (CROCO): NÃO EXECUTADOS.** A única produção CROCO é o piloto de 2 meses (nov–dez/2017). Com o split oficial (treino até 2018, validação 2019–2021, teste 2022–2024), esses modelos ficariam sem linhas de validação e teste. Rodar assim daria números sem sentido.
  - O roteiro está pronto: `--fonte croco` monta a escada E1–E5, o E6, as ablações de cada grupo de dinâmica e os controles. Ele roda sem mudança quando a série CROCO existir.
  - A decisão de produção (A: 2014–2024; B: 2000–2024; C: não produzir) está com o Tulio.
- **E7 (GLORYS) e controles: EXECUTADOS.** Os dados são a reanálise GLORYS12 mensal de 2000–2024 nos 1 790 pontos do ML: T e S de superfície, u e v e MLD.
  - **Classe:** REANÁLISE.
  - **Por que este é o controle certo:** o CROCO é forçado pelo GLORYS. Um ganho só poderá ser atribuído ao CROCO se ficar **acima** do que o GLORYS já dá.

## 1. Desenho e regras de justiça

**Referência:** o **E0_corrigido** (13.2B):
- features sem vazamento, hotspots H2;
- XGBoost 300/6/0,1, `scale_pos_weight` do treino;
- split 2001–2018 / 2019–2021 / 2022–2024;
- limiar de máximo F1 na validação.

O E0 prevê o mês T com dados até T−1 (13.3), então **toda feature física ou de satélite entra com defasagem de 1 mês, por data** (o valor de T−1).

**Iguais para todos os modelos:**
- as mesmas 500 900 linhas: o dataset do E0 sem os 4 pontos NAO_USAR_ESTUARIO_CANAL. Treino 372 456, validação 64 148, teste 64 296;
- o mesmo split, os mesmos hiperparâmetros e a mesma regra de limiar;
- **5 seeds.**

**Achado sobre as seeds.** Sem subamostragem, a seed do XGBoost não muda nada. Mas o resultado **depende da ordem das linhas de treino**: só embaralhar as linhas mudou a PR-AUC do E0 de 0,714 para 0,738.
- Por isso cada seed embaralha a ordem do treino, com a **mesma** permutação para todos os modelos.
- O desvio entre seeds fica entre 0,002 e 0,007.

**Controle observacional (adicionado).** O mesmo E0 com **SST e SSS de satélite** do mês T−1, que o ML já baixa (C1), e com clorofila T−1 (C2). Esse controle separa duas coisas: o ganho de "estado recente do mar" e o ganho específico da física de modelo.

**Significância:**
- diferença pareada de PR-AUC (as mesmas linhas, com a probabilidade média das 5 seeds);
- IC 95 % por bootstrap em **blocos espaço-temporais** (ano do alvo × região × trimestre = 36 blocos), com 500 reamostragens.

## 2. Tabela completa (teste 2022–2024, média de 5 seeds)

| modelo | features físicas / extras (T−1) | PR-AUC | ROC-AUC | F1 | recall | precisão | Brier | log loss | Δ PR-AUC vs E0 [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
| **E0_corrigido** | — | **0,722** | 0,896 | 0,678 | 0,689 | 0,668 | 0,118 | 0,429 | — |
| C1 E0 + satélite TS | SST, SSS de satélite | 0,786 | 0,905 | 0,707 | 0,745 | 0,674 | 0,100 | 0,348 | +0,064 [+0,011; +0,125] |
| C2 E0 + satélite TS + CHL | + clorofila T−1 | 0,758 | 0,911 | 0,702 | 0,643 | 0,773 | 0,096 | 0,347 | +0,036 [−0,022; +0,114] |
| E1g E0 + TS GLORYS | T, S de superfície | 0,787 | 0,907 | 0,707 | 0,730 | 0,685 | 0,104 | 0,359 | +0,065 [+0,015; +0,121] |
| **E2g E1g + MLD** | T, S, MLD | **0,812** | **0,917** | **0,723** | 0,696 | 0,753 | **0,090** | **0,311** | **+0,090 [+0,038; +0,148]** |
| **E7 E0 + GLORYS completo** | T, S, MLD, u, v, velocidade | 0,798 | 0,905 | 0,712 | 0,678 | 0,751 | 0,090 | 0,316 | +0,076 [+0,007; +0,144] |
| E7 − TS (ablação) | MLD, corrente | 0,770 | 0,896 | 0,686 | 0,647 | 0,731 | 0,107 | 0,365 | +0,048 [−0,017; +0,106] |
| E7 − MLD (ablação) | TS, corrente | 0,788 | 0,902 | 0,695 | 0,710 | 0,682 | 0,101 | 0,344 | +0,066 [+0,004; +0,130] |
| E7 − corrente (ablação) | TS, MLD (= E2g) | 0,812 | 0,917 | 0,723 | 0,696 | 0,753 | 0,090 | 0,311 | +0,090 [+0,038; +0,148] |
| E6g só GLORYS + tempo mínimo | mês, lat, lon + TS, MLD, corrente | 0,740 | 0,877 | 0,656 | 0,630 | 0,684 | 0,121 | 0,396 | +0,018 [−0,064; +0,116] |
| C3 E7 + satélite TS + CHL | tudo | 0,774 | 0,911 | 0,698 | 0,662 | 0,740 | 0,097 | 0,341 | +0,052 [−0,012; +0,127] |
| C4 C1 + física sem TS | satélite TS + MLD + corrente | 0,786 | 0,901 | 0,692 | 0,699 | 0,685 | 0,097 | 0,337 | +0,064 [−0,001; +0,129] |
| **C5 C1 + MLD GLORYS** | satélite TS + MLD | 0,801 | 0,914 | 0,721 | 0,732 | 0,710 | 0,092 | 0,320 | +0,079 [+0,028; +0,136] |

**Comparações pareadas decisivas** (`comparacoes_pareadas_extras.csv`):

| comparação | Δ PR-AUC | IC 95 % | bootstrap > 0 |
|---|---|---|---|
| E1g (TS GLORYS) × C1 (TS satélite) | +0,001 | [−0,020; +0,027] | 56 % |
| C5 (satélite + MLD GLORYS) × C1 | **+0,015** | [−0,001; +0,032] | 96 % |
| C4 (satélite + MLD + corrente) × C1 | +0,000 | [−0,029; +0,040] | 50 % |
| E7 (com corrente) × E2g (sem corrente) | −0,013 | [−0,037; +0,009] | 21 % |

## 3. Por horizonte (esquema de previsão 13.3, física e satélite em t0)

Δ PR-AUC contra o E0_forecast no teste, com IC 95 %, seed 42 (`por_horizonte.csv`):

| h | satélite TS | GLORYS TS + MLD | satélite + MLD GLORYS |
|---|---|---|---|
| 1 | +0,059 [+0,012; +0,115] | **+0,083** [+0,026; +0,135] | +0,080 [+0,028; +0,132] |
| 2 | +0,002 | **+0,064** [+0,029; +0,108] | +0,025 |
| 3 | −0,010 | +0,036 [+0,007; +0,086] | −0,001 |
| 4 | −0,038 | +0,040 [−0,024; +0,134] | −0,011 |
| 5 | −0,017 | +0,026 | −0,027 |
| 6 | −0,005 | −0,031 | −0,021 |
| 7 | −0,036 | −0,002 | −0,032 |
| 8 | −0,031 | +0,071 [+0,016; +0,126] | −0,005 |
| 9 | +0,003 | +0,023 | −0,005 |
| 10 | +0,061 [+0,025; +0,098] | +0,041 [+0,011; +0,098] | +0,058 |
| 11 | −0,002 | +0,003 | −0,004 |
| 12 | 0,000 | +0,010 | +0,034 |

O ganho do estado físico é **forte em h = 1–2 e some por volta de h = 5–6**. Ele ressurge em h = 8 e h = 10, provavelmente por coincidência sazonal entre os meses-alvo e de emissão (INFERÊNCIA).

## 4. Estratificações (E0 × melhores, teste)

`estratificacoes.csv`. Δ PR-AUC do E2g (TS + MLD GLORYS) contra o E0:

| estrato | onde ajuda | onde não ajuda ou piora |
|---|---|---|
| ano | 2022 +0,13; 2023 +0,08; 2024 +0,05 | — |
| estação | **SON +0,22**; JJA +0,06 | MAM +0,00; DJF +0,02 |
| mês | **set +0,26**; jun +0,11; out +0,09 | **fev −0,04**; mar −0,02; jul −0,01 |
| região | centro +0,10; sul +0,11 | norte +0,03 (o E7, com corrente: −0,02) |
| profundidade | 50–100 m +0,20; >100 m +0,11; 20–50 m +0,06 | < 20 m: −0,00 (costa rasa, onde o E0 já tem PR-AUC 0,99) |
| validade | D2_VALIDO +0,14 | NAO_VALIDADO_COSTEIRO −0,00 |

## 5. Respostas

1. **O CROCO melhora o ML?** **Ainda não sabemos: não foi testado**, porque não há série CROCO. O que o controle GLORYS mostra é isto:
   - **o estado físico recente melhora muito o E0** (+0,065 a +0,09 de PR-AUC);
   - **mas quase todo esse ganho vem de T/S do mês anterior, que o satélite já fornece** (C1: +0,064). T/S do GLORYS acrescenta +0,001 ao satélite.
   - **O que a física de modelo dá além do satélite é a MLD:** +0,015, IC [−0,001; +0,032], 96 % das reamostragens positivas. É um ganho no limite da significância.
   - Para o CROCO valer a pena, ele precisa superar o **C5** (satélite + MLD GLORYS), e não o E0.
2. **Qual grupo dá o maior ganho?** O **T/S de superfície recente** (+0,065), mas ele é **equivalente ao satélite**. Entre os grupos que só um modelo físico fornece, a **MLD** (+0,025 sobre o TS do GLORYS; +0,015 sobre o satélite).
3. **Algum grupo piora?** Sim:
   - a **corrente** (u, v, velocidade) piora o modelo com TS + MLD: −0,013, IC [−0,037; +0,009]. Isso aparece sobretudo no norte;
   - a **clorofila T−1** junto com o satélite também piora (C2 < C1), o que sugere instabilidade do modelo;
   - **só física, sem o histórico** (E6g), não supera o E0 de forma significativa.
4. **O ganho é estatisticamente consistente?**
   - O ganho do estado recente sobre o E0 é: IC > 0, 99–100 % das reamostragens positivas e estável entre seeds.
   - O ganho específico da física sobre o satélite **não é conclusivo**: só a MLD chega perto da significância.
   - Os IC são largos, porque o teste tem 3 anos e 36 blocos.
5. **Ganho geral ou sazonal/regional?** **Concentrado**: em **set–out (SON)**, no **centro e sul**, na **plataforma de 20–100 m** e em **2022**. É praticamente nulo em DJF e MAM, na costa rasa e no norte, e é negativo em fev, mar e jul. Por horizonte, só vale em **h = 1–2**.
6. **Qual modelo deve seguir para a versão final?**
   - Hoje, a referência para os testes do CROCO deve ser o **C1 (E0 + SST/SSS de satélite em T−1)**, ou o equivalente no esquema de previsão, e **não o E0 puro**. O E0 não tem estado recente, então qualquer física "ganha" dele sem mérito próprio.
   - A versão candidata é o **C5** (satélite + MLD), mas com ressalvas: ela depende de MLD, que no operacional virá do CROCO, porque o GLORYS só tem a reanálise com 1–2 meses de atraso; e o ganho está no limite da significância.
   - Quando houver série CROCO, rodar `--fonte croco` e exigir: CROCO_CORE > C5 (o GLORYS) e > C1 (o satélite), com IC > 0 em h = 1–2.
   - Antes da versão final, estabilizar o XGBoost (por exemplo, média de várias ordens ou subsample fixo), porque a ordem das linhas muda a PR-AUC em até ~0,02.

**Resultado negativo registrado:** com os dados de hoje, **não há evidência de que a física de modelo acrescente algo robusto além do estado observado por satélite**. A única exceção é a MLD, no limite da significância.
