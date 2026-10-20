# Mapa de Risco de Floração AlgaMar — relatório de fechamento

**Data:** 09/10/2026. **Ramo local:** `ML/repo`, `algamar/correcao-leakage`. Nada foi enviado ao GitHub da equipe.

**Para quem é:** para o Tulio apresentar e defender o resultado do projeto. Está escrito em português simples, mas todo número vem de uma execução registrada (a fonte de cada um está na seção 9).

**Novo nome do produto:** **Mapa de Risco de Floração AlgaMar** (id `algamar-risco-floracao`). O nome antigo falava em "Navier–Stokes", mas a física não entrou no modelo final. A seção 6 explica a escolha.

---

## 1. Resumo em cinco frases

1. A pergunta do projeto era: **simular o oceano com as equações de Navier–Stokes (modelo CROCO) melhora a previsão de florações de algas no litoral de São Paulo?**
2. A resposta, medida com controles, é **não**. As variáveis de movimento da água (correntes, vorticidade, divergência, transporte e velocidade vertical W) pioraram o modelo e ficaram abaixo até de um controle com os dados embaralhados. Temperatura, salinidade e camada de mistura deram, no máximo, um sinal pequeno e instável.
3. O que funciona é um modelo de **satélite + climatologia**. Ele acerta onde e em que mês costuma florescer (PR-AUC 0,78, ROC-AUC 0,90, de 1 a 12 meses à frente), validado em 4,7 anos que ele nunca viu.
4. Esse modelo **não é melhor que a climatologia** para prever um ano com mais ou menos floração que o normal. Por isso, o produto honesto é um **mapa de risco**, não um sistema de previsão de eventos.
5. Um resultado negativo bem medido é ciência válida: ele foi testado contra referências e controles, sem vazamento de dados, e evita gastar ~52 dias de máquina numa produção que não mudaria a conclusão.

## 2. O que foi feito

| parte | o que é | resultado |
|---|---|---|
| **Física (Navier–Stokes)** | Modelo oceânico regional CROCO com grade de 1,5 km, cobrindo todo o litoral de SP. É forçado pela reanálise GLORYS, pelo vento e calor do ERA5, por marés FES2014 e por 19 rios. | Validado contra medições da Marinha (BNDO) para **temperatura e salinidade**: erro de temperatura de 1,59 °C, menor que o do GLORYS (1,68 °C); salinidade 0,33. **Correntes não foram validadas** (não há medição de referência utilizável). |
| **Correção do ML da equipe** | O modelo original usava informação do futuro sem querer (vazamento): climatologias com todos os anos, média móvel que incluía o próprio mês e limiar escolhido no teste. | Corrigido. A nota honesta caiu de **0,80 para 0,74** de PR-AUC. |
| **Previsão de 12 meses** | Um modelo XGBoost que parte da climatologia de cada ponto e mês e aprende o desvio a partir do satélite. | PR-AUC 0,783 no teste (alvos de 2022-02 a 2026-09), validado em 1 084 740 previsões. |
| **Física no ML, teste longo** | A reanálise GLORYS (o mesmo oceano que alimenta o CROCO), com 25 anos, como substituta da física. | T/S: nada. Camada de mistura (MLD): +0,003. Correntes: pioraram. |
| **Física no ML, teste CROCO** | Piloto com **todo** o CROCO que existia: 41 dias em 4 meses parciais. | Correntes e dinâmica: pioraram, abaixo do embaralhado. T/S + MLD + N²: sinal fraco e instável. |

A produção CROCO de vários anos (2014–2024, ~52 dias de máquina) foi **descartada** pela regra de nada passar de 1–2 h de máquina por tarefa. A seção 4 mostra por que ela não mudaria a conclusão.

## 3. O resultado da física, variável por variável

**Como ler:** a PR-AUC mede o quanto o modelo põe as florações reais no topo da lista de risco (1 = perfeito). Δ é a diferença contra o modelo sem física. O IC 95 % é o intervalo de confiança: se ele cruza o zero, o ganho não é distinguível de acaso. O "controle embaralhado" mistura os valores físicos entre os pontos do mesmo mês; uma variável útil precisa ganhar dele.

| variável física | teste longo (GLORYS, 25 anos) | piloto CROCO (41 dias) | veredito |
|---|---|---|---|
| **correntes** (u, v, velocidade) | −0,013 [−0,037; +0,009] sobre o E0 | **−0,055** [−0,102; −0,015], pior que o embaralhado | **sem sinal; piora** |
| **vorticidade, divergência, transporte, W, SSH, HBL** | não existe no GLORYS mensal | **−0,061** [−0,123; −0,005], pior que o embaralhado; vorticidade, divergência e W com correlação \|ρ\| ≤ 0,05 | **sem sinal; piora** |
| **T/S de superfície** | −0,000 [−0,004; +0,004] no oficial | +0,032 [+0,003; +0,059] | o satélite já traz a mesma informação |
| **MLD** (camada de mistura) | **+0,003** [+0,0005; +0,006] no oficial | parte do CORE abaixo | ganho real, mas pequeno demais (+0,4 %) |
| **CORE** (T/S + MLD + N²) | — | +0,057 [+0,011; +0,090], ganha do embaralhado | **fraco e instável**: melhora 1 mês de 3 e piora os outros 2 |
| **T/S em profundidade** | — | +0,018 [−0,042; +0,057] | não significativo |

**Detalhe importante sobre o CORE:** o ganho conjunto vem quase todo de acertar a diferença de nível entre meses muito diferentes (fev/2014 com 52 % de floração contra dez/2017 com 3 %). São só 3 meses, e um deles tem 4 dias de simulação, sem aquecimento inicial do modelo (spin-up). Nesse mês, a temperatura do CROCO chega a contradizer a do satélite.

## 4. Por que este resultado negativo é ciência válida

**a. A hipótese foi testada de verdade, não descartada por opinião:**
- havia uma hipótese clara: a física melhora a previsão;
- havia uma medida definida antes: PR-AUC no teste, com IC por bootstrap em blocos (ano × região × trimestre);
- e havia uma decisão definida: só vale se superar o modelo sem física e o controle embaralhado.

**b. Os controles eliminam as explicações fáceis:**
- **sem vazamento:** a física entra só no mês de emissão, nunca depois; o split é por data; o limiar é escolhido na validação;
- **controle embaralhado:** correntes e dinâmica perderam até para dados misturados, então o que elas trazem é ruído;
- **controle de profundidade:** a profundidade sozinha não reproduz o ganho do CORE, então ele não é só "raso × fundo";
- **referências:** a climatologia e a persistência foram medidas no mesmo teste.

**c. Duas fontes independentes apontam para o mesmo lado.** O GLORYS, com 25 anos, e o CROCO, com 41 dias, concordam: as correntes não ajudam, e só a camada de mistura tem algum valor.

**d. Há explicações físicas plausíveis (INFERÊNCIA, não medidas aqui):**
- **escala de tempo:** o alvo é a clorofila **máxima do mês**, e o que domina esse número é o ciclo das estações. Correntes e vórtices mudam em dias; a média do mês apaga esse efeito;
- **informação repetida:** a temperatura do satélite já mostra a água fria de ressurgência. Por isso o T/S do modelo não acrescenta;
- **qualidade do dado:** as correntes do CROCO não puderam ser validadas, e o alvo vem de satélite perto da costa, onde a água turva atrapalha a leitura de clorofila.

**e. O custo foi controlado.** O piloto custou ~15 min de máquina e mostrou que uma produção de ~52 dias não mudaria a conclusão. Saber onde **não** investir é um resultado útil.

**O que este resultado NÃO diz:**
- não diz que a física do oceano é irrelevante para florações;
- diz que, **nesta escala** (mensal, 4 km, alvo de satélite) e **com estes dados**, ela não melhora a previsão;
- um alvo de campo (contagem de células), uma escala diária ou eventos costeiros dentro de baías poderiam dar outro resultado. Isso não foi testado.

## 5. O que de fato funciona

| item | valor (teste com alvos de 2022-02 a 2026-09, nunca vistos no treino) |
|---|---|
| dados de entrada | clorofila de satélite (Copernicus L4, 4 km), temperatura da superfície de satélite (OSTIA L4) e climatologia calculada só com anos anteriores |
| horizontes | 1 a 12 meses à frente (H1–H12), todos validados em hindcast |
| PR-AUC | **0,783** (climatologia 0,778; persistência 0,389) |
| ROC-AUC | 0,905 |
| acerto do nível ALTO | 79–81 % dos pontos marcados como ALTO tiveram floração |
| por horizonte | 0,792 em H1 → 0,777 em H12; quase não cai, porque a nota vem do padrão das estações |
| por região | sul (Peruíbe–Mongaguá) 0,89; centro (Santos) 0,80–0,81; norte (Ubatuba–São Sebastião) 0,65–0,68, a única com ganho de +0,02 a +0,03 sobre a climatologia |
| calibração | "20 %" acontece 26 % das vezes, "50 %" 49 % e "80 %" 85 % (levemente subconfiante) |
| software | um comando (`python algamar.py previsao`), modelos imutáveis com SHA256, log por execução, nunca inventa probabilidade quando falta dado; 108 testes passando |

**Leitura honesta:** o produto diz bem **onde e quando costuma florescer**. Ele **não** prevê, melhor que a média histórica, se um mês específico vai ter mais ou menos floração que o normal: em nenhum horizonte o ganho sobre a climatologia é significativo. A classificação é **HINDCAST VALIDADO**: ainda não houve nenhuma previsão em tempo real conferida depois.

## 6. Novo nome

**Escolhido: "Mapa de Risco de Floração AlgaMar".** O nome curto descreve o que o produto entrega:
- **mapa:** um valor por ponto de 4 km e por região;
- **risco:** uma probabilidade com níveis BAIXO/MODERADO/ALTO, não a previsão de um evento;
- **floração:** o alvo (CHL máxima mensal > 1,5 mg/m³).

A descrição completa vai junto: "a partir de clorofila e temperatura de satélite e da climatologia histórica". "Navier–Stokes" sai do nome porque a física não faz parte do modelo final.

**Onde foi aplicado:**
- `config/ml.toml`, nova seção `[produto]`: nome, id, descrição, fontes de dados e status;
- saídas do pipeline: `previsao.json` e `ULTIMA.json` agora têm o campo `produto` (schema 1.1.0). As colunas do Parquet não mudaram, então o dashboard e a API continuam funcionando;
- cartão dos modelos novos: status "HINDCAST_VALIDADO" e nome do produto (antes: "base de arquitetura, aguardando Navier–Stokes");
- `LEIAME_ML.md` e este relatório.

**O que não foi renomeado, de propósito:**
- os relatórios antigos (`RELATORIO_FINAL_ML_NAVIER_STOKES_ALGAMAR.md` e as etapas), que são histórico;
- a pasta `cfd_navier_stokes`, que é o projeto de física;
- o cartão do `baseline-hist-api-v1`, porque o modelo oficial é imutável (SHA256) e mudar o cartão quebraria o manifesto.

## 7. Versão enxuta: só os dados que se mostraram importantes

**O que saiu do modelo** (flag `use_features_redundantes = false`): sete itens do estado em t0 que repetem a informação das médias de 3 meses e das anomalias. Cada um tinha menos de 0,5 % da importância no **treino** do oficial (a escolha não olhou o teste).
- chl_max_t0;
- chl_media_t0m1 e chl_media_t0m2;
- sst_t0m1 e sst_t0m2;
- chl_max_anom_t0;
- sss_t0m1, que já estava desligada com a salinidade.

A versão enxuta fica com **19 features**, contra 25 da oficial.

**Grupos físicos bloqueados** (`[features].sem_sinal`): correntes CROCO, dinâmica CROCO, T/S GLORYS e a feature store Navier–Stokes legada.
- Se alguém ligar um deles, o treino recusa e explica o motivo.
- O código e os dados não foram apagados.
- `permitir_sem_sinal = true` religa tudo, só para reproduzir experimentos antigos.

**Ficam disponíveis, mas desligados:** CROCO T/S, CORE e vertical (sinal fraco, não confirmado) e MLD GLORYS (+0,003, sem fonte operacional no mês de emissão).

**Comparação no mesmo teste** (as duas versões lidas do registro, mesmas 1 084 740 linhas; script `fechamento_compara_enxuta.py`):

| conjunto | modelo | PR-AUC | ROC-AUC | Brier | F1 | Δ PR-AUC vs oficial [IC 95 %] |
|---|---|---|---|---|---|---|
| teste completo (1 084 740) | oficial (25 features) | 0,783 | 0,905 | 0,083 | 0,670 | — |
| | **enxuta (19 features)** | **0,786** | 0,906 | 0,082 | 0,675 | +0,003 [−0,002; +0,009] |
| | climatologia | 0,778 | 0,904 | 0,082 | — | −0,006 [−0,019; +0,007] |
| emissões 2022–2024 | oficial | 0,781 | 0,906 | 0,090 | 0,657 | — |
| | enxuta | 0,785 | 0,908 | 0,088 | 0,658 | +0,004 [−0,003; +0,013] |
| emissões 2025–26 (NRT) | oficial | 0,804 | 0,906 | 0,066 | 0,708 | — |
| | enxuta | 0,806 | 0,906 | 0,066 | 0,724 | +0,002 [−0,001; +0,004] |
| | climatologia | 0,815 | 0,922 | 0,068 | — | +0,011 [−0,007; +0,027] |

**Por horizonte:** a enxuta fica entre +0,001 (H1) e +0,004 (H10) acima da oficial em todos os 12 horizontes (0,792 → 0,781).

**Leitura:** tirar as 6 features redundantes **não perdeu nada**. A diferença é pequena e o IC cruza o zero, então as duas são **equivalentes**. A enxuta faz o mesmo com menos dados de entrada. Isso confirma que o que importa é: clorofila atual e média de 3 e 36 meses, anomalias de clorofila e SST, SST atual e de 3 meses, climatologia do mês-alvo e o ano anterior.

A versão enxuta foi registrada como **`risco-sat-clim-v1`**, uma pasta nova no registro. O `baseline-hist-api-v1` não foi alterado (mesmo SHA256). A seção 7b explica qual versão ficou como oficial.

## 7b. Litoral de SP inteiro (atualização de 09/10, à tarde)

**Problema encontrado:** a área herdada da equipe (lon −47,0 a −44,8) deixava de fora **Iguape, Ilha Comprida e Cananéia**, ~176 km de costa (18 % do litoral paulista). Medido contra a linha de costa oficial de SP (IBGE × GSHHG, 990 km): só 75 % da costa tinha um ponto do ML a até 10 km.

**O que foi feito:**
- **Área nova:** a área foi ampliada até a divisa com o Paraná (lon −48,2), mantendo a caixa da equipe (`config/ml.toml [area]`). A divisa SP/PR no mar é aproximada por uma reta perpendicular à costa a partir da Barra do Ararapira (`espacial.dentro_sp`); é uma APROXIMAÇÃO, não a linha legal.
- **Dados baixados (Copernicus):** clorofila, temperatura e salinidade diárias, 2000–2024 (MY/REP, 75 arquivos anuais, `baixar_litoral_sp_total.py`), e 2025-01 a 2026-09 (MY/NRT, 63 arquivos mensais).
- **Histórico refeito** com o mesmo método da equipe (`construir_historico_sp_total.py`). Nos 1 790 pontos da equipe, o resultado é **idêntico** ao CSV da equipe: diferença máxima de 2·10⁻⁶ mg/m³ na clorofila e 10⁻¹⁴ °C na temperatura (`resultados/sp_total/conferencia_equipe.json`).
- **Resultado:** **2 406 pontos**, contra 1 790. Agora **100 % da costa** tem um ponto a até 10 km, e 86 % a até 5 km. Cada município tem um ponto a ~2–5 km da costa.
- **Nova região** "extremo sul (Cananéia–Ilha Comprida–Iguape)", a oeste de −47,08.
- **Nova versão:** `risco-sat-clim-sp-v1`, a mesma receita enxuta treinada na área nova.

**Comparação no mesmo teste** (emissões ≥ 2022-01, h = 1…12; `compara_sp_total.py`, `resultados/sp_total/`):

| onde | modelo | PR-AUC | Δ [IC 95 %] |
|---|---|---|---|
| 1 790 pontos da equipe | oficial antigo (baseline-hist-api-v1) | 0,783 | — |
| | enxuta (risco-sat-clim-v1) | 0,786 | |
| | **SP total (risco-sat-clim-sp-v1)** | **0,790** | |
| | climatologia | 0,781 | |
| caixa da equipe (1 855 pontos, ver nota) | SP total × oficial antigo | 0,839 × 0,834 | +0,005 [−0,004; +0,015] |
| pontos novos do litoral sul (551) | **SP total** | **0,938** | +0,005 [+0,0002; +0,010] sobre a climatologia (0,933) |
| | oficial antigo (nunca viu esses pontos) | 0,930 | |
| litoral SP inteiro (2 406) | **SP total** | **0,880** | +0,007 [+0,002; +0,012] sobre a climatologia (0,873) |

**Por região** (SP total contra a climatologia, PR-AUC):
- **norte:** 0,766, Δ **+0,014** [+0,004; +0,024]. É o único ganho claro.
- **centro:** 0,844, Δ +0,006, não significativo.
- **sul:** 0,926, Δ +0,005, não significativo.
- **extremo sul:** 0,940, Δ +0,004, não significativo.

**Como ler:**
- A versão nova **não perdeu nada** nos pontos antigos; ficou até um pouco acima, mas dentro do acaso.
- A nota alta do litoral sul (0,94) vem de muitas células que **floram quase sempre** pelo satélite: 44 % de floração, contra 22 % na área antiga. São água turva de estuário e do Mar Pequeno, onde a clorofila de satélite é superestimada (caso 2, INFERÊNCIA). A climatologia já acerta 0,93 ali.
- Na caixa da equipe entraram **65 células costeiras** que o CSV da equipe não tinha. Elas floram 92 % dos meses, pelo mesmo motivo.
- **38 pontos** (31 deles nessa faixa costeira) **nunca** têm temperatura de satélite. A previsão os marca como DADOS_INSUFICIENTES, sem inventar probabilidade.

**Decisão:** o oficial passou a ser **`risco-sat-clim-sp-v1`**. É o único modelo treinado no litoral inteiro, e não piora nos pontos antigos. O `baseline-hist-api-v1` e o `risco-sat-clim-v1` ficam no registro, imutáveis; para voltar, basta trocar a linha `[modelo].oficial`. A primeira previsão com ele (t0 = 2026-09) saiu com status OK: 2 406 pontos, dos quais 2 368 com probabilidade.

**Arquivos antigos preservados:** o CSV da equipe, `painel_mensal.parquet` e `dados/brutos/` não foram alterados. O config agora aponta para `historico_sp_total_2000_2024.csv`, `painel_mensal_sp_total.parquet` e `dados/brutos_sp_total*`.

## 8. Limitações que continuam

1. Não supera a climatologia em anomalias, em nenhum horizonte.
2. Nenhuma previsão em tempo real foi conferida. Com os dados NRT de 2025–26, o modelo fica levemente abaixo da climatologia.
3. O alvo (CHL > 1,5 por satélite) nunca foi validado em campo e não é floração nociva confirmada.
4. Estuários, baías internas, o Mar Pequeno de Cananéia–Iguape e o Canal de São Sebastião não são confiáveis a 4 km: a clorofila de satélite em água turva é superestimada, e 38 pontos costeiros não têm temperatura.
5. As regiões são provisórias (4 faixas de longitude).
6. Nada foi enviado ao GitHub da equipe. A API e o site da equipe ainda servem o modelo antigo.

## 9. Fontes dos números

| números | arquivo |
|---|---|
| validação do CROCO (T 1,59 °C; S 0,33) | `cfd_navier_stokes/docs/etapa12_28_revalidacao_final.md`, `relatorio_final_algamar_fisica.md` |
| vazamento (0,80 → 0,74) | `docs/etapa13_02b_baseline_corrigido.md` |
| GLORYS no E0 e no oficial | `docs/etapa13_06_experimentos_e0_e7.md`, `docs/etapa13_07_glorys_no_oficial.md` |
| H1–H12, regiões, calibração | `docs/etapa13_07_validacao_12_meses.md` |
| piloto CROCO (41 dias) | `../RELATORIO_PILOTO_CROCO_ML.md`, `resultados/piloto_croco_ml/` |
| auditoria geral | `docs/RELATORIO_FINAL_ML_NAVIER_STOKES_ALGAMAR.md` |
| versão enxuta | `resultados/fechamento/` e `modelos/risco-sat-clim-v1/cartao.json` |

## 10. Perguntas prováveis e respostas curtas

- **"Então o Navier–Stokes foi perda de tempo?"** Não. Ele foi construído e validado para temperatura e salinidade (melhor que o GLORYS), e o teste mostrou com controles que, para este alvo, ele não acrescenta. Isso evita um investimento de ~52 dias de máquina sem retorno.
- **"41 dias não é pouco?"** É, e o relatório diz isso. Mas o teste de 25 anos com o GLORYS, que é o mesmo oceano que alimenta o CROCO, dá a mesma conclusão. E correntes e dinâmica perderam até para o controle embaralhado, o que não depende do tamanho da amostra para ser um mau sinal.
- **"Se o modelo é igual à climatologia, para que serve?"** Para entregar, de forma automática, auditável e atualizada todo mês, um mapa de onde e quando o risco é maior, com 4 km de resolução. Antes isso era digitado à mão no site.
- **"O que mudaria a conclusão?"** Um alvo de campo (contagem de células nocivas), uma escala diária ou dados dentro das baías. Nenhum deles cabe no escopo atual.
