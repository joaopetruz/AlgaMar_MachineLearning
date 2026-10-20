# Relatório final — Machine Learning e integração CROCO/Navier–Stokes do AlgaMar

**Data:** 2026-10-09. **Ramo local:** `ML/repo`, `algamar/correcao-leakage`, último commit 10ec698. Nada foi enviado ao GitHub da equipe.

**Natureza do documento:** é uma auditoria de consolidação. Não houve alteração de código nem de modelo nesta etapa.

**Rótulos de evidência:**
- MEDIDO = número saído de execução registrada;
- CÓDIGO = verificado no código;
- INFERÊNCIA = conclusão sem medição direta.

**Fontes:**
- revisão do repositório (08/10);
- etapas 13.2B, 13.3, 13.4, 13.5, 13.6, 13.7 (GLORYS e validação de 12 meses) e 13.8;
- física: etapas 12.x e 13.0–13.1 em `cfd_navier_stokes/docs/`.

---

## 1. Estado do ML

O modelo oficial é **`baseline-hist-api-v1`**, definido em `config/ml.toml [modelo].oficial` e registrado com SHA256 em `modelos/baseline-hist-api-v1/MANIFESTO.json`.

| item | valor |
|---|---|
| algoritmo | XGBoost multi-horizonte (um modelo para h = 1…12, com h como feature), com **base climatológica**: parte da log-odds da frequência histórica de floração do ponto/mês e aprende o desvio. 400 árvores, profundidade 6, lr 0,05, subsample 0,8, colsample 0,8, min_child_weight 50, seed 42. |
| features (25) | **posição/tempo:** h, mes_emissao, mes_alvo, latitude, longitude. **Estado em t0:** chl_media_t0, chl_max_t0, sst_t0, chl_media_t0m1, chl_media_t0m2, sst_t0m1, sst_t0m2, chl_media_3m, sst_3m, chl_media_36m e as anomalias de CHL média, CHL máx e SST. **Histórico do mês-alvo:** clim_chl_media_alvo, clim_chl_max_alvo, freq_floracao_alvo, n_anos_alvo, clim_sst_alvo, chl_max_alvo_ano_anterior, floracao_alvo_ano_anterior. **SSS desligada:** o NRT tem viés de +0,6 PSU (13.4). |
| target | floração = **CHL máxima mensal > 1,5 mg/m³** no ponto (satélite L4 4 km). É binário, por isso a métrica principal é a PR-AUC. |
| threshold | **0,39** (ALTO), escolhido como máximo F1 **na validação**; MODERADO ≥ 0,195. Nunca é escolhido no teste. |
| split | por data: treino com alvos até 2018-12; validação com emissões de 2019-01 e alvos até 2021-12; teste com emissões a partir de 2022-01 (alvos 2022-02 a 2026-09). Nenhum alvo é compartilhado entre conjuntos (teste unitário). |
| validação | hindcast em 1 084 740 previsões (ponto × emissão × h). Inclui comparação com climatologia e persistência, IC 95 % por bootstrap em blocos (ano × região × trimestre), calibração e estratificação por região e por ponto (13.7). |
| métricas no teste (MEDIDO, todos os h) | PR-AUC **0,783**, ROC-AUC 0,905, F1 0,670, recall 0,584, precisão 0,786, Brier 0,083, log loss 0,277. Climatologia: PR-AUC 0,778. |

**Outro modelo auditado: E0_corrigido.** É o modelo "sazonal" da equipe com o vazamento corrigido: 12 features, XGBoost 300/6/0,1, teste 2022–2024. Mede PR-AUC **0,737** (0,722 nas linhas sem pontos NAO_USAR, média de 5 seeds). Ele só vale para **1 mês à frente** (13.3). Fica como referência científica, não como modelo de previsão.

## 2. Leakage e problemas da revisão de 08/10

| # | problema | situação | evidência |
|---|---|---|---|
| 1a | climatologias (CHL, SSS) com todos os anos, inclusive teste e o próprio mês | **CORRIGIDO** | CÓDIGO: `climatologia_passada` usa só anos anteriores (cumsum + shift), em `features/sazonal.py:51-61` e `features/horizonte.py:77-89`. |
| 1b | média de 36 meses incluía o mês t | **CORRIGIDO** | CÓDIGO: `shift(1).rolling(36)` (`sazonal.py:70`). No esquema de previsão, a janela termina em t0. |
| 2 | limiar escolhido no teste; CV embaralhada | **CORRIGIDO** | CÓDIGO: `escolhe_limiar` só na validação (`pipeline.py:70`, `avaliacao.py:66`). A avaliação é por blocos temporais; os scripts antigos de CV embaralhada não são usados. |
| 3 | "ano anterior" por posição (`shift(12)`) | **CORRIGIDO** | CÓDIGO: `lag_por_data` (`sazonal.py:42`). No cubo, a defasagem é por índice de mês. |
| 4 | alvo sem justificativa (1,5 mg/m³) | **ABERTO** | É decisão científica, não vazamento. Mantido como DÚVIDA 1 desde 08/10. |
| 5 | hotspots de 2024 (ano de teste), em graus | **CORRIGIDO** | Decisão H2 do Tulio: recalculados com dados ≤ 2018, em km (`ANO_FIM_HOTSPOT = 2018`). O modelo oficial nem usa hotspot. |
| 6 | salinidade costeira preenchida por vizinho até 55 km | **CONTORNADO** | O modelo oficial **não usa SSS**. O CSV da equipe ainda contém o preenchimento; só o E0_corrigido (referência) usa SSS. |
| 7 | API da equipe serve um RF diferente do avaliado | **ABERTO no repositório da equipe** | O nosso pipeline serve só modelos do registro, com hash. O `api.py` e o `.pkl` da equipe não foram tocados. |
| 8 | `/predictions` não prevê nada | **RESOLVIDO no nosso pipeline; ABERTO na API da equipe** | `algamar.py previsao` emite t0 → t0+1…12 com features ≤ t0 (prova por perturbação na 13.3; assert de proveniência em cada execução). |
| 9 | previsão anual do site = climatologia digitada à mão | **ABERTO no site** | O site não foi alterado (regra do projeto). O nosso pipeline calcula as entradas sozinho. |
| 10 | ponto mais próximo em graus, CORS `*`, IP fixo | **ABERTO na API da equipe** | Fora do nosso escopo. O nosso código usa haversine (`espacial.py`). |
| 11 | xgboost e scipy fora do requirements | **CORRIGIDO** | CÓDIGO: `requirements.txt` com xgboost, scipy e pyarrow, com faixas de versão. |
| 12 | sem manifesto nem hash | **CORRIGIDO** | Manifesto do painel, MANIFESTO.json por modelo e hash em cada log de execução (13.8). |
| 13 | código espalhado em 3 ramos | **PARCIAL** | Tudo está consolidado no ramo local `algamar/correcao-leakage`, mas **não foi enviado** nem juntado ao `main` da equipe. |
| 14 | sem testes nem pipeline único | **CORRIGIDO** | `algamar.py` + `fluxo.py`; **106 testes** passando, 16 deles de ponta a ponta. |
| extra 13.2B | XGBoost sensível à ordem das linhas (até 0,02 de PR-AUC) | **MEDIDO, não corrigido** | Nos experimentos, as seeds permutam a ordem. O modelo oficial usa uma ordem fixa e é reprodutível bit a bit. |
| extra 13.8 | NaN de estado em t0 recebia probabilidade | **CORRIGIDO** | Ponto sem estado → probabilidade nula, DADOS_INSUFICIENTES (teste E2E). |
| extra 13.8 | `.gitignore` ignorava o código `algamar_ml/modelos/` | **CORRIGIDO** | Commit 10ec698. |

**Efeito medido do vazamento** no E0, teste 2022–2024: PR-AUC **0,800 → 0,737**. Os números antigos da equipe estavam inflados.

## 3. Previsão: gerar saída × previsão validada

O sistema **gera** saída para +1 a +12 meses, e **essa saída não usa nenhum dado do futuro**: o estado é sempre ≤ t0, verificado por teste e pelo assert de proveniência. Mas **gerar não é prever com habilidade**:

| horizonte | gera saída? | sem vazamento? | validado em hindcast? | melhor que a climatologia? (MEDIDO, 13.7) |
|---|---|---|---|---|
| +1 | sim | sim | sim | +0,009 de PR-AUC, IC [−0,006; +0,021]: **não significativo** |
| +2 | sim | sim | sim | +0,007, não significativo |
| +3 | sim | sim | sim | +0,006, não significativo |
| +4…+11 | sim | sim | sim | +0,004 a +0,007, não significativo |
| +12 | sim | sim | sim | +0,004, IC [−0,010; +0,020], **não significativo**; Brier pior que a climatologia |

**Conclusão:** o sistema prevê **o risco sazonal esperado** (onde e em que mês costuma florescer) com boa qualidade em todos os horizontes. **Não há evidência de que preveja anomalias**, ou seja, um mês ou ano com mais ou menos floração que o normal, em nenhum horizonte.

## 4. Localização

| item | valor |
|---|---|
| resolução | grade CHL L4 da Copernicus, **4 km** (~0,0417°) |
| pontos | **1 790** pontos oceânicos, lat −25,3 a −23,3 e lon −47,0 a −44,8 (litoral de SP, de Peruíbe a Ubatuba) |
| coordenadas | latitude e longitude do centro da célula, 5 casas decimais; distâncias em km (haversine) |
| regiões | 3 setores por longitude (provisórios, DÚVIDA 2): **sul** Peruíbe–Itanhaém–Mongaguá (lon < −46,6); **centro** Praia Grande–Santos–Guarujá–Bertioga (< −45,6); **norte** São Sebastião–Ilhabela–Caraguatatuba–Ubatuba |
| acerto de localização (MEDIDO) | AUC espacial de 0,95 por mês-alvo (climatologia 0,95); 79–81 % dos pontos ALTO floresceram |

**Limitações costeiras:**
- **Satélite:** o CHL L4 é gap-free, isto é, interpolado sob nuvens. Perto da costa, a água turva (caso 2) superestima clorofila (INFERÊNCIA, literatura de cor do oceano). Estuários, baías internas e o Canal de São Sebastião não são resolvidos a 4 km.
- **Física:** 12 pontos da grade (4 do ML) são **NAO_USAR_ESTUARIO_CANAL** e 146 são costeiros não validados. O GLORYS não tem o Canal de São Sebastião nem Ilhabela.
- **Dados in situ:** não há verdade de campo (HAB, contagem de células). O alvo é só satélite.

## 5. APIs: o que entra de fato na previsão

| dado | produto | classe | entra no modelo oficial? |
|---|---|---|---|
| clorofila | `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` (MY; também usado para 2025–26) | **observação** (satélite L4, reprocessada/interina) | sim: estado em t0, t0−1, t0−2, médias de 3 e 36 meses |
| clorofila NRT | `..._nrt_l4-gapfree...` | observação (tempo quase real) | catalogada; fallback se o MY atrasar |
| SST | `METOFFICE-GLO-SST-L4-REP-OBS-SST` até 2024; `...-NRT-OBS-SST-V2` em 2025–26 | **observação** (análise L4; REP = reprocessada) | sim: t0, t0−1, t0−2, média de 3 meses, anomalia |
| SSS | `cmems_obs-mob_glo_phy-sss_my/nrt` | observação | **não** (desligada por viés do NRT) |
| climatologia | calculada do próprio painel 2000–t0 | **climatologia** (passada) | sim: clim_*, freq_floracao_alvo, ano anterior |
| GLORYS12 mensal | `cmems_mod_glo_phy_my_0.083deg_P1M-m` | **reanálise** | **não** no oficial; só experimentos (flags `use_glorys_*`) |
| CROCO D2 | produção própria | **modelo regional forçado por reanálise** (GLORYS + ERA5) | **não**: só 2 meses produzidos |
| forecast (atmosfera/oceano) | — | **forecast** | **nenhum**: o sistema não usa previsões numéricas do futuro |

## 6. CROCO/Navier–Stokes

**Features integradas no código** (contrato 13.0, `croco_ml.py`; flags `use_croco_*` em `config/ml.toml`):

| grupo | features | status físico | testado no ML? |
|---|---|---|---|
| CROCO_TS | temperature_surface, salinity_surface | VALIDADA (T RMSE 1,59 °C; S 0,33 contra BNDO, 12.28) | **não**: só 2 meses de CROCO |
| CROCO_CORE | TS + mld, n2_max | VALIDADA / DERIVADA | não |
| CROCO_VERTICAL | T e S a 20 m, 50 m e no fundo | VALIDADA | não |
| CROCO_CURRENT | u, v, velocidade | NAO_VALIDADA_OBS | não |
| CROCO_DYNAMICS | ssh, w 10/50 m, vorticidade, ζ/f, divergência, hbl, transportes | NAO_VALIDADA_OBS | não |

**Teste com dado CROCO real:** só o piloto de 2 meses (nov–dez/2017). Ele passou de ponta a ponta no software (13.5), mas não permite treinar nem avaliar. A produção 2014–2024 (~52 dias de máquina) foi **descartada pelo Tulio** (regra: nada acima de 1–2 h).

**Proxy medido: GLORYS**, a reanálise que força o CROCO (13.6 e 13.7). Efeito sobre a PR-AUC:

| grupo (GLORYS) | sobre o E0_corrigido (13.6) | sobre o modelo oficial (13.7) | efeito |
|---|---|---|---|
| T/S de superfície (T−1) | +0,065 [+0,015; +0,121] | **−0,000** [−0,004; +0,004] | **melhorou** o E0, mas só porque o E0 não tem estado recente. O SST de satélite dá o mesmo (+0,064). **Não alterou** o oficial. |
| MLD | +0,025 sobre o TS; +0,015 sobre o satélite (limite) | **+0,003** [+0,0005; +0,006] | **melhorou pouco**: único ganho específico da física, pequeno |
| correntes (u, v, velocidade) | −0,013 [−0,037; +0,009] | não testado | **piorou** (sobretudo no norte) |
| só física, sem histórico (E6g) | +0,018 [−0,064; +0,116] | — | não alterou de forma significativa |

## 7. Comparação final (MEDIDO)

**A. Baseline corrigido (E0) × E0 + física GLORYS** (13.6). Teste 2022–2024, 64 296 linhas, 5 seeds:

| modelo | PR-AUC | ROC-AUC | F1 | recall | precisão | Brier | log loss | Δ PR-AUC [IC 95 %] |
|---|---|---|---|---|---|---|---|---|
| **E0_corrigido** | 0,722 | 0,896 | 0,678 | 0,689 | 0,668 | 0,118 | 0,429 | — |
| E0 + SST/SSS satélite T−1 (controle, sem física) | 0,786 | 0,905 | 0,707 | 0,745 | 0,674 | 0,100 | 0,348 | +0,064 [+0,011; +0,125] |
| E0 + GLORYS T/S + MLD | **0,812** | 0,917 | 0,723 | 0,696 | 0,753 | 0,090 | 0,311 | +0,090 [+0,038; +0,148] |
| E0 + GLORYS completo (com correntes) | 0,798 | 0,905 | 0,712 | 0,678 | 0,751 | 0,090 | 0,316 | +0,076 [+0,007; +0,144] |
| E0 + satélite + MLD GLORYS | 0,801 | 0,914 | 0,721 | 0,732 | 0,710 | 0,092 | 0,320 | +0,079 [+0,028; +0,136] |

**B. Modelo oficial × oficial + física GLORYS** (13.7). Teste com emissões 2022–2024, 773 280 linhas:

| modelo | PR-AUC | ROC-AUC | Brier | F1 | Δ PR-AUC [IC 95 %] |
|---|---|---|---|---|---|
| **oficial (hist-api-v1)** | 0,781 | 0,906 | 0,090 | 0,657 | — |
| + MLD GLORYS | 0,784 | 0,910 | 0,089 | 0,659 | +0,003 [+0,0005; +0,006] |
| + T/S + MLD GLORYS | 0,781 | 0,904 | 0,089 | 0,670 | −0,000 [−0,004; +0,004] |

**C. Baseline + CROCO:** **não há métricas**, porque não foi possível treinar sem a série CROCO. Não se inventa número.

## 8. Previsão de 12 meses: métricas por horizonte (MEDIDO, 13.7)

Modelo oficial, teste 2022-02 a 2026-09, 1 084 740 previsões:

| h | PR-AUC | ROC-AUC | F1 | recall | precisão | Brier | log loss | PR-AUC climatologia | PR-AUC persistência |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,792 | 0,912 | 0,678 | 0,596 | 0,786 | 0,080 | 0,267 | 0,783 | 0,511 |
| 2 | 0,795 | 0,912 | 0,681 | 0,594 | 0,796 | 0,081 | 0,269 | 0,788 | 0,458 |
| 3 | 0,792 | 0,909 | 0,678 | 0,592 | 0,794 | 0,081 | 0,272 | 0,786 | 0,409 |
| 4 | 0,788 | 0,906 | 0,674 | 0,588 | 0,790 | 0,082 | 0,274 | 0,783 | 0,390 |
| 5 | 0,784 | 0,903 | 0,672 | 0,585 | 0,787 | 0,083 | 0,278 | 0,779 | 0,372 |
| 6 | 0,780 | 0,902 | 0,668 | 0,583 | 0,783 | 0,083 | 0,279 | 0,775 | 0,366 |
| 7 | 0,781 | 0,903 | 0,667 | 0,580 | 0,786 | 0,083 | 0,279 | 0,775 | 0,373 |
| 8 | 0,778 | 0,902 | 0,666 | 0,579 | 0,783 | 0,083 | 0,277 | 0,772 | 0,362 |
| 9 | 0,776 | 0,902 | 0,664 | 0,580 | 0,777 | 0,083 | 0,278 | 0,769 | 0,360 |
| 10 | 0,776 | 0,901 | 0,664 | 0,579 | 0,779 | 0,084 | 0,282 | 0,772 | 0,364 |
| 11 | 0,777 | 0,901 | 0,664 | 0,576 | 0,785 | 0,085 | 0,284 | 0,772 | 0,398 |
| 12 | 0,777 | 0,900 | 0,664 | 0,576 | 0,783 | 0,086 | 0,286 | 0,773 | 0,389 |

**Leitura:**
- **Degradação:** pequena, de 0,792 (h1) para 0,777 (h12), porque a nota vem quase toda da climatologia.
- **Por região:** sul 0,89 e centro 0,80–0,81, sem ganho sobre a climatologia; norte 0,65–0,68, o único com ganho de +0,02 a +0,03.
- **Calibração:** "20 %" acontece 26 % das vezes, "50 %" 49 % e "80 %" 85 %. Levemente subconfiante.

## 9. Limitações (o que ainda impede uso real)

1. **Sem habilidade comprovada sobre a climatologia.** Em nenhum horizonte o ganho é significativo, e o Brier é até levemente pior. O produto é, na prática, risco climatológico.
2. **Nenhuma previsão em tempo real verificada.** Todo o teste é hindcast. 71 % das linhas usam satélite reprocessado (MY/REP), que não existe em tempo real.
3. **Inputs NRT não validados no modelo.** Em 2025–26 (CHL MY interina + SST NRT), o modelo fica levemente abaixo da climatologia (0,808 × 0,815 em h1).
4. **Navier–Stokes não testado no ML.** O CROCO só tem 2 meses, e a produção plurianual foi descartada. O proxy GLORYS mostra ganho de no máximo +0,003 no oficial.
5. **Alvo sem validação de campo.** "CHL máx > 1,5" vem de satélite (gap-filled, com viés costeiro), sem justificativa formal (DÚVIDA 1). Não é floração nociva (HAB) confirmada.
6. **Regiões provisórias:** 3 setores por longitude (DÚVIDA 2).
7. **Costa e estuários fora do alcance confiável:** 4 km de satélite; pontos NAO_USAR e costeiros não validados.
8. **Calibração subconfiante.** A recalibração não foi feita.
9. **Sem operação:** não há agendamento mensal, monitoramento nem arquivo de previsões congeladas para verificação.
10. **Instabilidade do XGBoost à ordem das linhas** (~0,02 de PR-AUC entre ordens). O oficial é reprodutível, mas o número tem essa incerteza.
11. **Integração com a equipe pendente:** nada foi enviado ao GitHub. A API e o site da equipe ainda servem o RF antigo, com `/predictions` sem previsão real e o formulário climatológico manual.

## 10. Status

**C — hindcast científico funcional.**
- **Por que C:** o fluxo completo, sem vazamento, reprodutível, versionado, logado e testado gera previsões H1–H12 e foi validado em 4,7 anos de hindcast.
- **Por que não B:** não houve nenhuma previsão em tempo real verificada, os inputs NRT não foram validados e a habilidade não supera a climatologia.
- **Por que não D:** não é protótipo; o pipeline é completo e testado.

## 11. Próximos passos (só os realmente restantes)

1. **Decidir o posicionamento do produto (Tulio):** publicar como "risco climatológico de floração ajustado pelo estado atual" ou parar aqui. Sem novo dado preditivo, mais modelagem não deve mudar a conclusão (INFERÊNCIA, apoiada por 13.3, 13.6 e 13.7).
2. **Piloto, se aprovado:**
   - agendar `python algamar.py previsao --baixar` uma vez por mês, poucos minutos por execução;
   - arquivar `saidas/<run_id>`;
   - conferir contra o observado quando cada mês-alvo chegar, por pelo menos 3–6 meses.
3. **Recalibração isotônica** na validação. É de minutos.
4. **Entregar à equipe:**
   - enviar o ramo;
   - substituir o `/predictions` e o formulário do site pelo `saidas/ULTIMA.json` / `previsao.parquet`.
   - Precisa da autorização do Tulio e da equipe.
5. **Fechar as DÚVIDAS 1 (alvo) e 2 (regiões)** com o Tulio e a equipe.

Não estão listados: produção CROCO plurianual (descartada pelo custo), novos modelos de ML e novas features físicas. Os experimentos mostram que eles não mudariam a conclusão dentro do limite de 1–2 h.

---

## RESUMO PARA O TULIO

- **O que foi feito:**
  - Corrigimos o vazamento do ML da equipe (a nota caiu de 0,80 para 0,74, o valor honesto).
  - Criamos um modelo de previsão de 1 a 12 meses e testamos se a física (CROCO e GLORYS) ajuda.
  - Validamos tudo em 4,7 anos de dados que o modelo não viu e montamos um pipeline único, com logs, versões e testes (106 passando).
- **O ML funciona?** Sim, como software: roda de ponta a ponta, sem vazamento, e não inventa probabilidade quando falta dado. As notas são altas (PR-AUC ~0,78, ROC ~0,90).
- **Prevê o futuro de verdade?** Só em parte. Ele acerta **o padrão de sempre**: onde e em que época do ano costuma florescer. Mas não consegue prever, melhor que essa média histórica, se um mês vai ter **mais ou menos** floração que o normal.
- **Prevê 12 meses?** Gera os 12 meses, e eles têm quase a mesma qualidade, justamente porque a qualidade vem do padrão sazonal. H1 e H12 são praticamente iguais à climatologia.
- **Consegue dizer onde?** Sim, a 4 km: 79–81 % dos pontos marcados como ALTO floresceram. O melhor setor é o sul (Peruíbe–Mongaguá); o pior, o norte (Ubatuba–São Sebastião), que é também o único onde o modelo ganha um pouco da média histórica. Estuários e o canal não são confiáveis.
- **O Navier–Stokes ajudou?** O CROCO não pôde ser testado no ML: produzir 11 anos levaria ~52 dias de máquina, e você vetou. A reanálise GLORYS, que alimenta o CROCO, deu **+0,003** no modelo oficial, praticamente nada. A única variável útil foi a profundidade da camada de mistura; as correntes atrapalharam.
- **Quais métricas melhoraram?**
  - Contra o ML original corrigido (0,72): o estado recente do mar subiu a nota para 0,79–0,81.
  - O modelo oficial chegou a 0,78, e a física acrescentou só +0,003.
  - Contra "repetir a média histórica", nenhuma melhora significativa.
- **Problemas que ainda existem:**
  - o modelo não supera a climatologia;
  - nada foi testado em tempo real;
  - a definição de floração (CHL > 1,5 por satélite) nunca foi validada em campo;
  - a costa e os estuários ficam fora;
  - a API e o site da equipe ainda mostram o modelo antigo.
- **Podemos considerar o projeto finalizado?**
  - **Como pesquisa e hindcast:** sim, a parte técnica está fechada e documentada.
  - **Como sistema de previsão operacional:** não. Ele pode virar um "mapa de risco climatológico mensal", se você quiser publicar assim, depois de um piloto de alguns meses conferindo as previsões.

**Parado. Aguardando a sua autorização para qualquer próximo passo.**
