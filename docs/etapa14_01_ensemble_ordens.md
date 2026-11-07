# Etapa 14.1 — Ensemble de ordens e recalibração isotônica

Data: 2026-10-09. Ramo local `algamar/ensemble-ordens`, criado a partir de `algamar/correcao-leakage` no commit 51b13f6. Nada foi enviado.

Script: `etapa14_01_ensemble_ordens.py`. Saídas em `resultados/etapa14_01_ensemble/` e log em `resultados/etapa14_01_ensemble.log`. O experimento levou 33 min de máquina, mais 5 min para registrar a versão bruta.

## Resumo

| pergunta | resposta |
|---|---|
| A variância entre ordens caiu? | **Sim.** O desvio-padrão da PR-AUC de teste caiu de 0,0008 (40 modelos simples) para 0,0002 (5 ensembles de 8), cerca de 4× menor. A amplitude caiu de 0,0039 para 0,0006. |
| A calibração melhorou? | **Não com a isotônica.** Ela piora o teste: ECE 0,0159 no oficial contra 0,0277 na v2 isotônica. O ensemble **bruto** melhora um pouco: ECE 0,0119. |
| A v2 supera o oficial? | **Não.** v2 isotônica: Δ PR-AUC −0,0080 [−0,0109; −0,0051] e Δ Brier +0,0016. Ensemble bruto: Δ PR-AUC −0,0004 [−0,0020; +0,0011], um empate. |
| Supera a climatologia? | O ensemble bruto e o oficial superam por pouco: +0,0065 a +0,0069 de PR-AUC; o IC do oficial é [+0,002; +0,011]. A v2 isotônica não supera: −0,0012 [−0,005; +0,003]. |
| Foi promovida? | **Não.** O critério era IC do Δ PR-AUC > 0 e Brier não pior, e nenhuma v2 o cumpre. `[modelo].oficial` continua `risco-sat-clim-sp-v1`. |

## O que foi implementado

`algamar_ml/modelos/ensemble.py`:
- **Seeds derivadas, nunca reutilizadas.**
  - `SeedSequence(seed_mestra).spawn(K)` gera um filho por membro.
  - Cada filho faz `spawn(2)`, e cada uma das duas chaves gera uma seed: uma para a permutação das linhas de treino, outra para o XGBoost.
  - As 16 seeds e as spawn_keys ficam gravadas no cartão (`ensemble.seeds`) e em `modelo.json`.
  - Aumentar K não muda os membros já existentes.
- **Funções puras.** `treina_ensemble` não altera o DataFrame recebido (há teste para isso), e `com_calibracao` devolve um objeto novo.
- **Membros.** Cada um usa os hiperparâmetros de `[modelo]` (400 árvores, profundidade 6, lr 0,05, subsample 0,8, colsample 0,8, min_child_weight 50), a base climatológica e early stopping na validação. O ensemble faz a média das probabilidades.
- **Isotônica opcional.** É ajustada só na validação, e depois o limiar é re-escolhido na validação (máximo F1). O teste nunca entra nessas escolhas.
- **Registro.**
  - `modelo.json` é um descritor, e os membros ficam em `membro_XX.json`.
  - O MANIFESTO cobre os membros, e um membro alterado gera `ErroRegistro`.
  - `tr.probabilidade` e o fluxo de previsão funcionam sem mudança.
  - Os modelos simples antigos continuam carregando como antes.
- **Config e linha de comando.** Seção `[ensemble]` (`ativo = false`, `k = 8`, `seed_mestra = 42`, `calibracao = ""`; a v2 isotônica foi treinada com `"isotonica"`) e a opção `python algamar.py treinar --versao X --ensemble`.
- **Testes.** 11 novos em `tests/test_ensemble.py`; a suíte completa tem 120 testes, todos passando.

Versões novas registradas. Nenhuma versão existente foi tocada.

| versão | descrição |
|---|---|
| `risco-sat-clim-sp-v2-ens` | 8 ordens, seed mestra 42, isotônica da validação, limiar 0,38 |
| `risco-sat-clim-sp-v2-ens-bruto` | os mesmos 8 membros, **bit a bit idênticos** (mesmo SHA256, o que confirma a reprodutibilidade), sem calibração, limiar 0,48 |

## Comparação no mesmo teste

Condições:
- Emissões ≥ 2022-01, h = 1..12, painel SP total, 1 458 036 linhas.
- O oficial foi lido do registro, sem retreino.
- IC 95 % por bootstrap em blocos de ano do alvo × região × trimestre, com 200 reamostragens, como na 13.7.

| h | oficial | v2 isotônica | ens. bruto | climatologia | Δ bruto − oficial [IC 95 %] | Δ oficial − clim [IC 95 %] |
|---|---|---|---|---|---|---|
| 1 | 0,8852 | 0,8765 | 0,8841 | 0,8757 | −0,0012 [−0,0032; +0,0010] | +0,0095 [+0,0034; +0,0151] |
| 2 | 0,8863 | 0,8776 | 0,8851 | 0,8783 | −0,0012 [−0,0034; +0,0007] | +0,0080 [+0,0042; +0,0130] |
| 3 | 0,8845 | 0,8764 | 0,8839 | 0,8776 | −0,0006 [−0,0021; +0,0010] | +0,0069 [+0,0024; +0,0108] |
| 6 | 0,8785 | 0,8704 | 0,8781 | 0,8720 | −0,0004 [−0,0019; +0,0009] | +0,0065 [+0,0012; +0,0113] |
| 9 | 0,8760 | 0,8685 | 0,8762 | 0,8696 | +0,0002 [−0,0016; +0,0021] | +0,0064 [+0,0008; +0,0118] |
| 12 | 0,8768 | 0,8688 | 0,8764 | 0,8700 | −0,0004 [−0,0026; +0,0021] | +0,0068 [+0,0024; +0,0118] |
| **todos** | **0,8801** | **0,8720** | **0,8797** | **0,8732** | **−0,0004 [−0,0020; +0,0011]** | **+0,0069 [+0,0019; +0,0114]** |

A tabela completa, com os 12 horizontes, está em `resultados/etapa14_01_ensemble/comparacao_por_horizonte.csv`. Na v2 isotônica:
- Δ PR-AUC contra o oficial: de −0,0074 a −0,0087 em todos os horizontes, com IC sempre abaixo de zero.
- Δ Brier: de +0,0013 a +0,0018, com IC cruzando zero.

Outras métricas (todos os horizontes):

| | oficial | v2 isotônica | ens. bruto | climatologia |
|---|---|---|---|---|
| ROC-AUC | 0,9337 | 0,9329 | 0,9328 | 0,9297 |
| Brier | 0,0824 | 0,0840 | **0,0823** | 0,0837 |
| log loss | **0,2718** | 0,2793 | 0,2721 | 0,2785 |
| F1 (limiar da validação) | 0,7732 (0,48) | 0,7731 (0,38) | 0,7730 (0,48) | — |
| ECE no teste | 0,0159 | 0,0277 | **0,0119** | 0,0123 |
| prob. média / freq. obs. | 0,279 / 0,270 | 0,243 / 0,270 | 0,274 / 0,270 | 0,281 / 0,270 |

## Variância entre ordens

| | n | PR-AUC média | desvio-padrão | mín–máx | amplitude |
|---|---|---|---|---|---|
| modelo simples (1 ordem) | 40 | 0,8793 | 0,0008 | 0,8779–0,8817 | 0,0039 |
| ensemble de 8, bruto | 5 | 0,8797 | **0,0002** | 0,8796–0,8802 | **0,0006** |
| ensemble de 8, isotônica | 5 | 0,8712 | 0,0008 | 0,8699–0,8720 | 0,0022 |
| oficial (ordem original) | 1 | 0,8801 | — | — | — |

- Por horizonte, o desvio-padrão cai de 0,0007–0,0012 para 0,0002–0,0003 (`variancia_por_horizonte.csv`). A queda é perto do √8 esperado.
- No painel SP total, a dependência da ordem é menor do que a 13.6 mediu no E0 da área da equipe: amplitude de 0,004, não de ~0,02.
- O oficial (0,8801) fica acima da média dos 40 modelos simples (0,8793). Parte dos +0,0004 que ele tem sobre o ensemble bruto é sorte da ordem original, e o ensemble bruto dá o valor esperado.

## Por que a isotônica piorou

1. **Mudança na taxa de floração entre validação e teste.** A validação (alvos 2019–2021) tem 24,5 % de floração e o teste (2022–2026) tem 27,0 %.
   - A isotônica ajustada na validação puxa as probabilidades para baixo. Média: 0,277 → 0,245.
   - No teste, o modelo passa a subestimar: prevê 0,243 contra 0,270 observado.
   - Na faixa de 10–20 %, a frequência observada passa a ser 22 %.
2. **Perda de ordenação.** A função em degraus tem só 94 valores distintos e cria empates. A PR-AUC cai até na própria validação: 0,8862 → 0,8809.
3. **O diagnóstico de subconfiança não se confirma no painel SP total.** No teste, o oficial é levemente **superconfiante**: prevê 0,279 contra 0,270 observado, e a climatologia também (0,281). Não há subconfiança para corrigir.

## Recomendação (para o Tulio decidir)

- **Não promover** nenhuma das duas v2, porque nenhuma cumpre o critério.
- O `risco-sat-clim-sp-v2-ens-bruto` é uma alternativa **equivalente em habilidade** ao oficial. Comparado a ele:
  - PR-AUC: Δ −0,0004, IC cruza zero.
  - Brier: 0,0823 contra 0,0824.
  - ECE: 0,0119 contra 0,0159, ou seja, um pouco mais bem calibrado.
  - É 4× mais estável entre ordens: o número publicado não depende da sorte da ordem das linhas.
  - Custa 8× mais tempo de treino (~5 min contra ~40 s) e de previsão (desprezível).

  Se a reprodutibilidade valer esse custo, ele é o candidato. Trocar é uma linha: `[modelo].oficial = "risco-sat-clim-sp-v2-ens-bruto"`.
- Recalibração isotônica: manter **desligada** neste modelo enquanto validação e teste tiverem taxas de floração diferentes. Se voltar a ser testada, vale experimentar uma calibração com termo de nível (Platt) ou uma validação mais recente.

## Reproduzir e voltar atrás

- Rodar `python etapa14_01_ensemble_ordens.py`. As probabilidades dos membros ficam em cache em `dados/etapa14_01/`, e as versões já registradas são lidas do registro, não retreinadas.
- Treinar um ensemble novo: `python algamar.py treinar --versao NOVA --ensemble`, usando a seção `[ensemble]` do config (padrão agora `calibracao = ""`, a variante bruta; para a isotônica, `calibracao = "isotonica"`).
- O oficial **não mudou**: `[modelo].oficial = "risco-sat-clim-sp-v1"`. As duas v2 são versões novas e imutáveis. Para descartá-las, basta não usá-las.
- Os modelos antigos simples carregam exatamente como antes: o manifesto deles não muda, e há teste para isso.

## Troca do oficial (2026-10-09, confirmada pelo Tulio)

- `[modelo].oficial` passou de `risco-sat-clim-sp-v1` para **`risco-sat-clim-sp-v2-ens-bruto`**, por decisão do Tulio, embora o critério automático de promoção não tenha sido cumprido. A habilidade é a mesma (Δ PR-AUC −0,0004, com IC cruzando zero). A troca foi feita pela reprodutibilidade e pela calibração um pouco melhor.
- Conferência:
  - os 120 testes passam;
  - `python algamar.py previsao --hoje 2026-10-09` terminou com status OK (run `20261009T174913Z_891248`, emissão t0 = 2026-09, alvos de 2026-10 a 2027-09);
  - foram geradas 28 872 linhas (2 406 pontos × 12 horizontes); 38 pontos ficaram DADOS_INSUFICIENTES, os que nunca têm SST;
  - o modelo foi carregado do registro com os SHA256 dos 10 arquivos conferidos contra o MANIFESTO;
  - log em `resultados/previsao_v2_ens_bruto_2026-10-09.log`.
- **Para voltar:** no `config/ml.toml`, use `oficial = "risco-sat-clim-sp-v1"`. A versão continua no registro, imutável.
