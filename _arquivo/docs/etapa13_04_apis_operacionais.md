# Etapa 13.4 — APIs operacionais: histórico + ambiente atual (sem Navier–Stokes)

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage`. Nada foi enviado ao GitHub e o CROCO não entrou.

**Código:**
- `algamar_ml/operacional.py`: catálogo classificado, disponibilidade em t0, validação, fallback e proveniência;
- `pipeline_operacional.py`: o fluxo completo;
- `tests/test_operacional.py`: 27 testes.

**Saídas:**
- `resultados/etapa13_04/`: `catalogo_fontes.csv`, `cobertura_catalogo_2026-10-09.json` (consulta ao vivo do catálogo Copernicus) e `features_disponiveis_em_t0.csv`;
- `previsoes/operacional_2026-09/`: previsão e `relatorio_operacional.json`.

---

## 1. Inventário das fontes

`catalogo_fontes.csv`. A cobertura foi consultada no catálogo da Copernicus em 09/10/2026.

| fonte | variável | unidade | res. temporal | res. espacial | período no catálogo | atualização | classe | uso no ML |
|---|---|---|---|---|---|---|---|---|
| CHL L4 gap-free **MY** | clorofila | mg m⁻³ | diária | 4 km | 1997-09-04 → **2026-10-01** | dias a semanas (parte interina) | ANÁLISE (L4, obs.) | treino + inferência |
| CHL L4 gap-free **NRT** | clorofila | mg m⁻³ | diária | 4 km | 2026-09-30 → 2026-10-07 (janela móvel) | ~1 dia | ANÁLISE | reserva |
| OSTIA SST **REP** | temperatura | K → °C | diária | 0,05° | 1981-10-01 → **2026-03-31** | ~6 meses | ANÁLISE (L4, obs.) | treino |
| OSTIA SST **NRT** | temperatura | K → °C | diária | 0,05° | 2024-01-17 → **2026-10-07** | ~1 dia | ANÁLISE | **inferência** |
| SSS multi-obs **MY** | salinidade | PSU | diária | 0,125° | 1993-01-01 → **2024-12-15** | 1–2 anos | ANÁLISE | treino |
| SSS multi-obs **NRT** | salinidade | PSU | diária | 0,125° | 2024-01-01 → 2026-10-03 | ~1 semana | ANÁLISE | **não usada**: viés +0,60 PSU, r 0,61 contra o MY |
| GLORYS12 **MY** | T, S, u, v, ζ 3D | vários | diária | 1/12° | 1993-01-01 → 2026-08-25 | 1–2 meses | **REANÁLISE** | futuro: forçante do CROCO |
| GLORYS12 **ANFC** | correntes etc. | vários | diária | 1/12° | 2022-06-01 → **2026-10-18** | diária | **FORECAST**, cerca de 10 dias à frente | não usado (horizonte curto demais) |
| Vento L4 **NRT** | vento | m s⁻¹ | horária | 0,125° | 2024-06-13 → 2026-10-07 | ~1 dia | ANÁLISE | não usado (DÚVIDA 3, sem histórico longo) |
| ERA5 / ERA5T | atmosfera | vários | horária | 0,25° | 1940 → hoje − ~5 dias | ~5 dias | **REANÁLISE** | futuro: forçante do CROCO |
| Open-Meteo archive | precipitação | mm | diária | ~0,1–0,25° | — | ~5 dias | **REANÁLISE** | removida (83d5f2b) |
| HABdata (eventos) | floração | evento | evento | ponto | 2016–2024 | irregular | **OBSERVAÇÃO REAL** | só validação externa |
| climatologias do ML | derivadas | — | mensal | 4 km | do painel | — | **CLIMATOLOGIA** | treino + inferência (só anos anteriores) |
| CROCO D2 (Navier–Stokes) | física 3D | vários | diária → mensal | 1,5 km | piloto nov–dez/2017 | depende das forçantes | **REANÁLISE/hindcast** (modelo forçado por GLORYS/ERA5) | futuro |

Todas as fontes Copernicus estão em lat/lon WGS84. A caixa do ML é lon −47,0 a −44,8 e lat −25,3 a −23,3. T e S são baixadas com margem de 0,5° e interpoladas para a grade CHL.

## 2. Classificação dos tipos

- **OBSERVAÇÃO REAL:** só os eventos HABdata, que não entram no treino. CHL, SST e SSS **não são observações pontuais**: são campos L4 diários, com lacunas preenchidas por interpolação.
- **ANÁLISE:** CHL L4, OSTIA SST, SSS multi-obs e vento L4. São produtos de satélite (+ in situ) interpolados, sem modelo dinâmico. As versões MY/REP são **reprocessadas** e as NRT são **de tempo quase real**.
- **REANÁLISE:** GLORYS MY, ERA5, Open-Meteo archive (vem de reanálises) e o CROCO forçado por elas.
- **FORECAST:** GLORYS ANFC além de hoje, só ~10 dias.
- **CLIMATOLOGIA:** as climatologias e frequências do próprio ML, calculadas só com anos anteriores.

**Consequência:** um sistema só é operacional se usa, no mês de emissão, **apenas** versões NRT ou a parte interina do MY já publicada naquela data. Produtos REP/MY que chegam meses depois só servem para treino e hindcast.

## 3. Informação disponível na data da previsão

Um exemplo concreto: previsão emitida em **09/10/2026** com t0 = **set/2026**, o último mês completo.

| variável em t0 | produto na data | disponível? |
|---|---|---|
| clorofila (set/2026) | CHL MY, parte interina, que vai até 01/10 | **sim** |
| SST (set/2026) | OSTIA **NRT** (o REP só vai até mar/2026) | **sim** |
| SSS (set/2026) | só NRT (o MY termina em 2024-12) | existe, mas **inconsistente**: desligada |
| vento | NRT | existe, sem histórico compatível para treinar |
| física 3D | GLORYS MY até 25/08; ANFC até 18/10 | parcial; ainda não é feature |

`features_disponiveis_em_t0.csv` traz todas as 34 features possíveis: a regra temporal, o mês mais recente usado em h = 1 e h = 12 e as fontes da emissão de set/2026.

- **Todas usam só meses ≤ t0, em todos os horizontes.** Um teste por h garante isso.
- O modelo em uso (`baseline-hist-api-v1`) **não usa salinidade**.
- O `e0-forecast-v1` depende de salinidade (ano anterior, climatologia e média de 36 meses). Para operar depois de 2024 ele precisaria de SSS NRT, que hoje é inconsistente. Esse é um motivo a mais para **não** operar o E0 como está.

## 4. Histórico × tempo real

| | treino / hindcast | inferência real |
|---|---|---|
| clorofila | histórico da equipe (MY, 2000–2024) + CHL MY 2025+ | CHL MY (parte interina) do mês t0; o NRT fica de reserva |
| SST | OSTIA REP (até 2024 da equipe; REP até mar/2026) | OSTIA **NRT** (viés +0,07 °C contra o REP, r 0,998: compatível) |
| SSS | SSS MY (até 2024-12) | **nenhuma** (NRT desligada) |
| climatologias | anos anteriores à linha | todos os anos já observados (todos ≤ t0) |
| alvo | CHL máximo mensal > 1,5 | não existe na emissão: é o que se prevê |

O painel (`dados/processados/painel_mensal.parquet`) guarda `fonte_clorofila`, `fonte_temperatura` e `fonte_salinidade` por mês. Assim fica explícito quando o treino viu REP e a inferência usa NRT.

## 5. Pipeline

`pipeline_operacional.py` (um comando):

```
api_ingestion       catálogo Copernicus (cobertura de cada produto; falha = INDISPONIVEL registrado)
                    --baixar: atualizar_dados_atuais.py (MY se cobre o mês inteiro, senão NRT) + construir_painel.py
      ↓
validation          cada arquivo diário: fuso UTC, tempos únicos e crescentes, dentro do mês, >= 80 % dos dias
      ↓
temporal_alignment  mês civil UTC; variável de arquivo reprovado → NaN no mês, com fonte "ANULADO"
      ↓
spatial_alignment   T e S interpoladas (bilinear) na grade CHL; checa caixa, alinhamento com os 1 790 pontos,
                    duplicatas, unidades (K × °C), faixa física, ausentes e cobertura por região
      ↓
feature_generation  constroi_linhas(t0) + proveniência por feature (variável, meses usados, produtos)
      ↓
forecast            12 meses × 1 790 pontos; colunas de qualidade: emissao_status, atraso_meses,
                    regiao_status, estado_t0_completo
```

**Execução real** (09/10/2026, sem download novo):
- os 6 produtos responderam no catálogo;
- 86 de 87 arquivos brutos foram aprovados;
- o 1 reprovado é o SST NRT de jan/2024, com 48 % dos dias (o produto começa em 17/01). Ele só serviu à checagem de consistência e não entra no painel, então nada foi anulado;
- a emissão saiu em **set/2026** com status OK e atraso 0;
- 21 480 linhas, de out/2026 a set/2027, sem ponto com estado incompleto.

## 6. Fallback (nada é inventado)

| situação | reação |
|---|---|
| API indisponível | catálogo marcado INDISPONIVEL; segue com os dados locais, e a emissão recua se o mês não existir |
| arquivo com < 80 % dos dias, tempo fora do mês, duplicado ou fuso ≠ UTC | variável **anulada** (NaN) naquele mês, com fonte "ANULADO" |
| NaN em alguns pontos (≤ 50 %) | mantém NaN (o XGBoost trata ausentes), `estado_t0_completo = False`, AVISO na validação |
| variável do modelo > 50 % ausente, ou mês ausente | o mês **não serve**: recua até 3 meses |
| região com < 50 % dos pontos com dado | previsão emitida, com `regiao_status = INSUFICIENTE` |
| atraso de atualização | t0 = último mês válido; `ATRASADO` (1–2 meses) ou `DESATUALIZADO` (> 2); nenhum mês válido nos últimos 4 → `SEM_EMISSAO` |
| unidade errada (ex.: K) ou valor fora da faixa física | FALHA na validação: o mês não é usado |

## 7. Proveniência

Cada feature registra a variável, o intervalo de meses usado e o produto de cada mês (`relatorio_operacional.json`). Exemplos da emissão de set/2026:

- `sst_t0` → `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2 (NRT)`, mês 2026-09;
- `chl_media_t0` → `cmems_obs-oc_…_my_l4-gapfree-multi-4km_P1D (MY)`, 2026-09;
- `chl_media_36m` → CHL MY 2025–26 + histórico da equipe (MY/REP até 2024-12), 36 meses de 2023-10 a 2026-09;
- `sst_anom_t0` → SST NRT (t0) + SST REP + histórico (climatologia);
- `latitude`, `h` → fixos (grade/configuração).

## 8. Testes

`tests/test_operacional.py` tem 27 testes. A suíte completa tem **79 passando**. Eles cobrem:

| tema | testes |
|---|---|
| timestamp e fuso | UTC aceito; fuso America/Sao_Paulo recusado; tempo fora do mês e duplicado recusados |
| duplicatas | chave (ponto, ano, mês) repetida → FALHA |
| coordenadas | ponto fora da caixa → FALHA; ponto deslocado da grade CHL → FALHA |
| unidades | SST em K detectada |
| faixa física | salinidade 55 → FALHA |
| ausentes | 20 % → AVISO; 100 % → FALHA; região sem dados → INSUFICIENTE |
| atualização temporal | mês faltando → recua e marca ATRASADO; dado velho → DESATUALIZADO ou SEM_EMISSAO; variável ausente → recua sem preencher |
| disponibilidade em t0 | as 34 features usam só meses ≤ t0 em h = 1…12 |
| proveniência | fonte por feature (NRT × MY) e meses usados |
| classificação | ANFC = FORECAST, ERA5 = REANÁLISE |

## 9. Respostas

1. **Quais APIs podem ser usadas operacionalmente?**
   - CHL L4 (MY interino, com o NRT de reserva) e OSTIA SST NRT, que já estão em uso.
   - Vento L4 NRT existe, mas não tem histórico compatível para treinar.
   - A SSS NRT existe, mas está inconsistente e por isso desligada.
   - O GLORYS ANFC é operacional, mas só cobre ~10 dias.
2. **Quais são só reanálises?** GLORYS MY, ERA5 (o ERA5T sai com ~5 dias e pode servir de "estado atual" para forçar o CROCO), Open-Meteo archive e o CROCO forçado por elas.
   - OSTIA REP e SSS MY não são reanálises: são análises reprocessadas, que servem só para treino por chegarem tarde.
3. **O que estará disponível na data da previsão?** Clorofila e SST do mês t0 e de todos os meses anteriores, além das climatologias. Salinidade só até 2024-12 (MY). Vento NRT, se for adotado. Física (GLORYS e CROCO) só com o atraso das forçantes.
4. **O pipeline recebe dados novos automaticamente?** **Parcialmente.** Um único comando (`pipeline_operacional.py --baixar`) consulta o catálogo, baixa os meses novos, refaz o painel, valida, decide a emissão e prevê. Mas:
   - **não há agendamento** (precisa ser disparado);
   - depende do login Copernicus já configurado na máquina;
   - não há alerta automático;
   - a API `/forecast` lê a pasta de previsão gerada, e não dispara o pipeline.
5. **O que precisa ser corrigido:**
   - agendamento mensal (por volta do dia 5 a 10) com alerta quando o status for diferente de OK;
   - checagem automática de consistência NRT × MY a cada nova sobreposição, para detectar mudança de versão de produto;
   - decisão sobre a SSS (fonte consistente, ou a do CROCO);
   - escolher o modelo oficial de previsão (pendente da 13.3; o E0 depende de SSS);
   - definir as regiões (DÚVIDA 2);
   - retreino periódico quando o MY/REP substituir o NRT, registrando versão e fonte;
   - ligar `/forecast` à saída `operacional_<mês>`.
