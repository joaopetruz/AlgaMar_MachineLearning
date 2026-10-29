# Índice de condições propícias: validação (v2, 24/10/2026)

Gerado por `indice_condicoes.py`. Regras e fontes: [fichas_especies_condicoes.md](fichas_especies_condicoes.md).

## O que foi feito

1. **Séries mensais 2000–out/2026** por município de SP (raio de 25 km) e por três zonas do Sul onde as florações de *Dinophysis* foram documentadas: PR (Paranaguá), SC norte (Penha/Itajaí) e SC centro (Florianópolis), com raio de 40 km.
   - Fontes: reanálise GLORYS emendada com o sistema operacional do Copernicus. Comparadas nos 51 meses em comum, em Santos, as duas têm correlação de 0,98 na temperatura, 0,67 na salinidade e 0,77 na camada de mistura; a diferença média foi corrigida.
   - Também entram vento L4 (com NRT a partir de jul/2026) e clorofila de satélite.
   - Outubro de 2026 é a média dos dias 1 a 23 de análise mais os dias seguintes da previsão de 10 dias do sistema.
2. **Índice mensal:** fração ponderada das condições da ficha satisfeitas. Fora da época da espécie, o nível máximo é "parcialmente propício" (escolha do projeto).
3. **Conferência com as florações conhecidas de *Dinophysis*** (Mafra Jr. et al. 2026):
   - principais: outono de 2008 (abr–jun), abr–jul de 2016 e jun–jul de 2024;
   - secundárias: 2011–2014, 2021 e 2022. São menos intensas, ou o texto do artigo é ambíguo quanto ao local.
4. **Previsão de 1 a 3 meses:**
   - salinidade e temperatura pela persistência da anomalia (regressão no próprio histórico);
   - camada de mistura, vento e clorofila pela frequência do mês no histórico.

## Resultado da conferência (*Dinophysis*, zonas do Sul)

Na **v1** havia 5 condições. A **v2** acrescentou a 6ª, a **mudança brusca**: a água esfriou 1 °C ou mais, ou a salinidade caiu 0,5 ou mais, em relação ao mês anterior. Os artigos descrevem "quedas bruscas de salinidade" e frentes frias; os limiares redondos são escolha do projeto.

| zona | meses mar–set em "propício" (v1 → v2) | eventos principais "propício" | eventos com pelo menos "parcialmente propício" (v2) | índice médio nos eventos (v2) | índice médio em mar–set sem evento (v2) |
|---|---|---|---|---|---|
| PR (Paranaguá) | 13,8% → 7,9% | 2/9 | 9/9 | 0,60 | 0,42 |
| SC norte | 12,7% → 10,6% | 4/9 | 6/9 | 0,59 | 0,42 |
| SC centro | 11,6% → 11,6% | 4/9 | 4/9 | 0,56 | 0,40 |

**Leitura:**
- O índice fica mais alto nos meses de floração.
- Em "propício", acerta 10 dos 27 meses de evento, nas duas versões. A v2 dá menos alarme falso: cerca de 10% dos meses de mar–set, contra cerca de 13% na v1.
- Em "parcialmente propício ou mais", a v2 pega 19 dos 27 meses de evento; a v1 pegava 23.
- Ficou a v2, porque um alarme que acende menos e acerta o mesmo vale mais.

**Quais condições estavam presentes nas florações** (média das três zonas):

| condição | nos meses de floração | em mar–set sem floração | serve para distinguir? |
|---|---|---|---|
| água mais fria que o normal (≤ −1 °C) | 63% | 8% | **sim, é o sinal mais forte** |
| mudança brusca (v2) | 41% | 19% | sim |
| vento de sul | 81% | 67% | um pouco |
| salinidade baixa | 33% | 21% | um pouco |
| coluna estratificada (MLD) | 41% | 43% | **não** |

A salinidade e a estratificação aparecem fracas por causa do dado, não da biologia. Os artigos medem quedas de 6 a 8 em poucas semanas, numa faixa estreita junto à costa, e a média mensal de um modelo de 9 km não enxerga isso direito. Os pesos **não** foram reajustados com esses números: três eventos são poucos, e o ajuste viraria "decorar" 2008, 2016 e 2024.

## Resultado da previsão (o que o dashboard mostraria antes do evento, v2)

| horizonte | índice previsto antes dos eventos | índice previsto em mar–set sem evento |
|---|---|---|
| 1 mês | 0,430 | 0,424 |
| 2 meses | 0,408 | 0,435 |
| 3 meses | 0,409 | 0,437 |

**Leitura honesta:** um mês antes, a previsão **não** distinguiu os meses de floração. A água fria e a queda de salinidade que acompanharam as florações chegaram de repente, com a frente fria e a pluma, e não estavam no mês anterior. A previsão de 1 a 3 meses vale como **tendência da época** ("estamos entrando no período de risco"). O índice que funciona é o **do mês corrente**, com os dados quase em tempo real.

## Conferência com episódios de SP (24/10/2026)

Pela primeira vez, há eventos **do litoral de SP** para conferir o índice. As fontes são a Resolução Conjunta SES/SIMA/SAA-1/2021, o CEBIMar/USP e o Instituto de Pesca.

**Junho de 2016, *Dinophysis*:** nota da CETESB de 06/06/2016 para Santos, São Sebastião e Peruíbe, com interdição de moluscos no 2º semestre.

| mês | Santos | Peruíbe | São Sebastião |
|---|---|---|---|
| abr/2016 | 0,26 | 0,26 | 0,16 |
| mai/2016 | 0,52 | 0,52 | 0,35 |
| **jun/2016** | **0,68** | **0,68** | **0,68** |
| jul/2016 | 0,68 | **1,00** | 0,26 |
| ago/2016 | **0,84** | **0,84** | 0,26 |

- O índice **subiu justamente na hora certa**: em junho, os três municípios ficaram em "parcialmente propício", no limite de "propício".
- Junho de 2016 em Santos ficou entre os 6% maiores índices de todos os 189 meses de mar–set (só 11 foram maiores).
- Em junho, estavam presentes a água fria, o vento sul, a mudança brusca e a floração no Sul. A salinidade baixa e a estratificação não apareceram.
- Desde esta versão, os municípios de SP têm a condição **"propício no Sul neste mês ou no anterior"**, porque o plano de SP começa o Alerta 1 com floração em SC/PR. Ela acende em 18,5% dos meses de mar–set. Com ela, os meses "propício" em SP caíram de 6,2% para 5,0%, e junho de 2016 subiu de 0,64 para 0,68.

**Janeiro–fevereiro de 2025, maré vermelha no Litoral Norte:** manchas de 12 a 27/01, causadas por *Mesodinium rubrum*, que não é tóxico e é a presa de *Dinophysis*. Nas coletas de 4–5/02, não havia ficotoxinas nos moluscos.
- *Dinophysis*: "desfavorável", fora da época (0,18–0,54).
- ***Pseudo-nitzschia*: "propício" (1,00) em Ubatuba, Caraguatatuba, Ilhabela e São Sebastião.** O CEBIMar relata ressurgência de água fria no verão de 2025, e o índice a viu (condição de nutrientes acesa). Mas não houve toxina nos moluscos. Fica registrado como **alarme sem floração tóxica**: "propício" não é previsão de floração, e é exatamente por isso que o índice não diz que vai florescer.
- *G. catenatum*: "propício" (0,81) em fev/2025, também sem toxina nos moluscos.

**Todos os episódios tóxicos de SP (Tabela 02 da resolução, DOE de 24/04/2021, 2000–2018):**

| episódio | alga | índice |
|---|---|---|
| jun–jul/2016, Baixada Santista e Litoral Norte | *Dinophysis acuminata* | jun: 0,68 em 9 dos 11 municípios e 0,52 em Ilhabela e Ubatuba; jul: 0,84–1,00 de Praia Grande a Peruíbe |
| abr/2018, São Sebastião | *Dinophysis acuminata* | 0,48 (parcialmente propício; percentil 82 do município) |
| mai/2018, Ubatuba | *Dinophysis acuminata* | 0,32 (desfavorável; percentil 35) |
| set/2017, Litoral Norte | *Gymnodinium* (?) | 0,50 (parcialmente propício; fora da época fev–jun) |
| fev/2000 e nov/2017 | *Trichodesmium* | (cianobactéria, fora do índice) |

*Dinophysis* em SP, comparando os 24 meses-município com floração aos 2.622 outros meses de mar–set:

| | meses com floração | outros meses de mar–set |
|---|---|---|
| índice médio | **0,60** | 0,38 |
| "propício" | **17%** | 5% |
| pelo menos "parcialmente propício" | **79%** | 45% |

Condições presentes nos meses com floração e nos outros:
- **floração no Sul: 92% contra 18%**;
- **água fria: 67% contra 10%**;
- mudança brusca: 54% contra 20%;
- vento sul: 88% contra 70%;
- estratificação: 29% contra 44%, ou seja, não ajuda.

**Leitura:**
- O índice funciona bem para floração **regional que vem do Sul**: em 2016, acendeu nos dois meses e em quase todo o litoral.
- Para florações **locais**, como as de 2018 no Litoral Norte, quase não acendeu. Nelas não havia água fria, vento sul nem floração no Sul, e a média mensal de um modelo de 9 km não enxerga o que acontece dentro de uma baía.
- 22 dos 24 meses com floração são de 2016, então os números refletem sobretudo esse evento.

**Março de 2019, canal de São Sebastião** (CEBIMar/USP, Ciotti, Flores e Migotto, 03/04/2019): maré vermelha em 13–14/03, com mais de 2 milhões de células/L. Havia *Alexandrium* e *Dinophysis* no dia 13, e no dia 14 era quase só *Margalefidinium*. A água estava acima de 29 °C, com salinidade baixa (~33). Quanto menos sal, mais células: os autores apontam nutrientes trazidos pela água de rio.

| espécie | fev/2019 | **mar/2019** | abr/2019 |
|---|---|---|---|
| *Dinophysis* | 0,10 | **0,58** (parcialmente propício) | 0,26 |
| *G. catenatum* | 0,62 | **0,81** (propício) | 0,81 |
| *Pseudo-nitzschia* | 1,00 | 1,00 | 0,69 |
| *A. catenella* | 0,00 | 0,17 | 0,17 |

- Desta vez, **o dado de satélite viu a água doce**: o modelo marca salinidade de 34,1, 1,2 abaixo do normal do mês, contra ~33 medidos. A temperatura do modelo foi 28,1 °C, 2,7 acima do normal, contra mais de 29 medidos.
- O *Dinophysis* subiu de 0,10 para 0,58 por causa da salinidade baixa, da mudança brusca e do vento sul.
- O *Alexandrium* desse evento estava em água de **29 °C**, ou seja, não é o *A. catenella* de água fria do índice. Isso confirma que o litoral de SP tem *Alexandrium* de água quente, que o índice **não cobre**.

**Chuva local (estação meteorológica do CEBIMar/USP, São Sebastião, jun/2013–jun/2025):**
- Em todos os 4 episódios de dinoflagelados do Litoral Norte com dados da estação, choveu 50% ou mais acima do normal no mês ou no mês anterior:
  - jun/2016: +225%;
  - abr/2018: +56%;
  - mai/2018: mês anterior +56%;
  - mar/2019: +172%, e fev +265%.
- Nos outros meses de mar–set, isso acontece em 45% das vezes. Os dois eventos sem chuva foram o *Gymnodinium* (?) de set/2017 (−52%) e o *Mesodinium* de jan/2025, que não é tóxico.
- Desde esta versão, *Dinophysis* tem a condição **chuva forte local** (peso 0,6), só nos 4 municípios do Litoral Norte. Ela fica sem valor quando a estação não tem dado (depois de jun/2025) e, nesse caso, sai da conta.

| *Dinophysis* em SP (mar–set) | meses com floração (26) | outros meses (2.620) |
|---|---|---|
| índice médio | 0,62 | 0,38 |
| "propício" (antes → agora) | 15% → **23%** | 4,9% → 5,0% |
| pelo menos "parcialmente propício" (antes → agora) | 81% → **92%** | 44,7% → 43,8% |

Atenção: a condição foi escolhida **depois** de olhar esses mesmos 4 eventos. Esses números são otimistas até ser conferidos com eventos novos.
- Ubatuba em mai/2018 continua "desfavorável" (0,38).
- A direção do vento da estação não foi usada, porque a série muda de comportamento em ago/2021, provavelmente na troca de estação.
- Os dados ficam só no computador (`dados/cebimar/`, fora do Git): a fonte não permite reproduzir os dados brutos.

**Antes de florações em SP:** em jan/2014, houve *Mesodinium* em Ilhabela, presa de *Dinophysis*, mas sem *Dinophysis* tóxico registrado depois. O índice de *Dinophysis* de Ilhabela ficou "desfavorável" de jan a jul/2014 (0,16–0,48), coerente com nenhum evento.

## *Pseudo-nitzschia* (ficha v2, 24/10/2026)

- **v1:** no litoral de SP, a temperatura passa de 17 °C e a salinidade de 30 o ano inteiro. O índice ficava "propício" em 58% dos meses, sempre de out a abr, e virava um calendário.
- **v2:** com os artigos enviados (Mafra et al. 2006 e 2024; Braga 1999, de Ubatuba), a condição de **chegada de nutrientes** ganhou peso alto. Ela é satisfeita quando entra água fria da ACAS (1 °C ou mais abaixo do normal), quando a clorofila sobe 25% ou quando a coluna se mistura forte (percentil ≥ 70). Isso acontece em 54% dos meses de out–abr em SP.
- Distribuição em SP: 24% dos meses "desfavorável", 44% "parcialmente propício" e 32% "propício". O índice agora varia de mês para mês.
- Ainda não há como conferir: o OBIS tem só 9 registros, de 2 meses.
- Na previsão, a chance de nutrientes vem da frequência do mês no histórico. Por isso, os meses de verão aparecem como "propício" (0,8–0,9): a época, a temperatura e a salinidade já estão garantidas, e há cerca de 55% de chance de chegarem nutrientes.

## *Gymnodinium catenatum* (ficha v2, 24/10/2026)

Fontes: Tavares et al. 2009, Schramm et al. 2006, Band-Schmidt et al. 2019 e Méndez e Ferrari 2003 (Uruguai). As condições são época de fev a jun, água de 17 a 29 °C, influência de água doce e entrada de água fria.

| evento conhecido | índice | nível |
|---|---|---|
| Paranaguá, fev/2003 | 0,62 | parcialmente propício |
| Paranaguá, mar/2003 | 0,62 | parcialmente propício |
| Penha, mar/2000 | 0,62 | parcialmente propício |
| Penha, abr/2006 | 0,62 | parcialmente propício |
| Penha, mai/2006 | 0,81 | propício |

- Nos meses de fev–jun sem evento, o índice médio é 0,68, e 25% dos meses ficam em "propício". **O índice ainda não separa as florações.**
- A condição de água doce não aparece em nenhum evento: o modelo marca salinidade acima de 34, contra 31–32 medidos nas baías.
- O esfriamento apareceu em 20% dos eventos, contra 18% nos outros meses.
- Não houve ajuste para acertar os eventos.

## *Alexandrium catenella* (incluída em 24/10/2026)

- Florações de campo só com água de 9–15 °C (Uruguai, Chile, Tasmânia) e coluna estratificada. A água fria é **obrigatória**: sem ela, o nível máximo é "desfavorável".
- A superfície do mar de SP nunca chegou a 9–16 °C de 2000 a 2026, então o índice fica **"desfavorável" o tempo todo em SP**. Nas zonas de SC, isso aconteceu em 0,6% dos meses.
- Pela literatura, o risco dessa espécie está no RS e no Uruguai. Esse índice não cobre os *Alexandrium* de água quente do Brasil (*A. minutum*, *A. tamiyavanichi*), que não têm dados de campo.

## Previsão atual (gerada em 24/10/2026, `saidas/condicoes_propicias.json`, v2)

- ***Dinophysis*:** fim da época.
  - Out/2026: "parcialmente propício" nos 14 municípios. Santos tem o índice mais alto (0,64), com a salinidade 2,4 abaixo do normal.
  - Nov/2026 a jan/2027: "desfavorável" na maioria dos municípios, e "parcialmente propício" de Bertioga a Iguape em novembro.
- ***Pseudo-nitzschia*:** out/2026 "parcialmente propício" (0,69, sem sinal de nutrientes no mês); nov/2026 a jan/2027 "propício" (0,80–0,88), pela época de verão.
- ***Gymnodinium catenatum*:** "parcialmente propício" de out/2026 a jan/2027 (0,46–0,69), fora da época dela (fev–jun).
- ***Alexandrium catenella*:** "desfavorável" em todos os municípios (água quente demais).

No dashboard: página **Condições Propícias** (`software/pages/condicoes-propicias.html`), que lê `GET /ml/condicoes-propicias` (API do ML: `/condicoes-propicias`).

## Limites e próximos passos

- As regras de *Dinophysis* vêm do Sul (25–28° S). No litoral de SP (23–25° S), falta conferir com dados locais.
- O índice diz que o mar está favorável, não que a alga está lá. Os registros do OBIS são só de presença e não medem alarme falso.
- Para melhorar:
  1. a queda brusca em dados **diários** (a v2 usa a variação de um mês para o outro);
  2. vazão dos rios ou chuva;
  3. a previsão sazonal do C3S (temperatura do mar e vento, até 6 meses; precisa de conta no CDS);
  4. dados de monitoramento (CETESB, CIDASC ou EPAGRI) para calibrar de verdade.
