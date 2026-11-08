# Governança e LGPD (Etapa 4)

> Plano de proteção de dados, base legal da decisão de crédito e retenção.

Este documento descreve o tratamento de dados pessoais **neste projeto** e o
plano que uma fintech precisaria executar para operá-lo em produção. Onde a
distinção importa, ela está explícita: o que **está implementado aqui** vs. o
que **seria necessário em produção**.

## 1. Inventário de dados

### O que o modelo usa

Dez atributos, todos econômico-financeiros ou demográficos agregados:

| Atributo | É dado pessoal? | É sensível (art. 5º, II)? |
|---|---|---|
| `age` | **sim** | não |
| `NumberOfDependents` | **sim** | não |
| `MonthlyIncome` | **sim** | não |
| `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines` | sim (derivado) | não |
| contadores de atraso (3 colunas) | sim (histórico de crédito) | não |
| `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines` | sim | não |

**Nenhum dado pessoal sensível** na definição do art. 5º, II da LGPD: não há
origem racial ou étnica, convicção religiosa, opinião política, filiação
sindical, dado referente à saúde, à vida sexual, genético ou biométrico.

### O que o modelo não usa, por decisão

Não há **nenhum identificador direto** no pipeline: sem CPF, nome, endereço,
telefone, e-mail ou número de conta. O dataset público já vem assim, e o
projeto mantém essa propriedade por construção.

Isso é aplicado tecnicamente, não por convenção: `strict=True` nos schemas
Pandera **recusa qualquer coluna não declarada no contrato**. Um upstream que
anexe um CPF ao lote tem a ingestão bloqueada. O teste
`test_contrato_rejeita_coluna_desconhecida` e a injeção `coluna_inesperada`
(que usa justamente uma coluna `cpf`) garantem que a regra não regrida.

É *privacy by design* no sentido literal: a arquitetura torna o vazamento de
PII uma falha ruidosa em vez de um descuido silencioso.

> **Limite honesto.** Os atributos usados são pseudonimizados, não anonimizados.
> Renda, idade, número de dependentes e histórico de crédito combinados podem
> ser reidentificáveis. Continuam sendo dados pessoais sob a LGPD, e todo o
> regime de proteção se aplica.

## 2. Base legal (art. 7º)

Para análise de crédito, duas bases se combinam:

**Art. 7º, V — execução de contrato ou procedimentos preliminares.** A análise
de risco é condição prévia à concessão: sem avaliar, não há como contratar. É
a base principal para quem **solicitou** crédito.

**Art. 7º, IX — legítimo interesse.** Sustenta o monitoramento contínuo de
desempenho do modelo e a prevenção à inadimplência. Exige **teste de
proporcionalidade** documentado (finalidade legítima, necessidade, balanço com
os direitos do titular) e registro no RIPD.

Observação de contexto: no cadastro positivo brasileiro (Lei 12.414/2011, com a
redação da LC 166/2019), o histórico de pagamentos compõe o escore de crédito
sob regime próprio, com direitos específicos de acesso e revisão. Uma operação
real deve ancorar o tratamento também nessa lei, além da LGPD.

**Consentimento (art. 7º, I) não é a base adequada aqui** — e usá-lo seria um
erro comum. Consentimento precisa ser livre e revogável; numa relação em que
negar o tratamento inviabiliza o produto, ele não é livre. Invocá-lo criaria
uma base frágil e um direito de revogação que a instituição não consegue
honrar sem encerrar o contrato.

## 3. Princípios aplicados (art. 6º)

| Princípio | Como é cumprido neste projeto |
|---|---|
| **Finalidade** | Dados usados só para decisão e monitoramento de crédito. |
| **Adequação** | Todos os dez atributos têm relação direta com capacidade de pagamento. |
| **Necessidade** | O contrato limita o escopo ao mínimo: `strict=True` barra qualquer coluna extra. |
| **Livre acesso** | `GET /monitoring/summary` e o dashboard expõem como o modelo decide; em produção, um canal ao titular. |
| **Qualidade dos dados** | É a Etapa 1 inteira: contrato, portão e laudos auditáveis. |
| **Transparência** | Limiares versionados em `settings.py`; decisão e threshold devolvidos em toda resposta da API. |
| **Segurança** | Container não-root, dependências travadas no lockfile, CodeQL e zizmor no CI. |
| **Prevenção** | Alertas de drift e o portão de dados existem para evitar dano antes que ocorra. |
| **Não discriminação** | Ver seção 5. |
| **Responsabilização** | Linhagem completa no MLflow: dado → treino → modelo → decisão. |

## 4. Decisão automatizada e direito à revisão (art. 20)

O art. 20 dá ao titular o direito de **solicitar revisão de decisões tomadas
unicamente com base em tratamento automatizado** que afetem seus interesses —
e crédito negado afeta.

**Implementado aqui:**

- a API devolve, em toda resposta, a probabilidade estimada, a decisão, o
  limiar aplicado e qual modelo decidiu — nada de caixa-preta sem rótulo;
- o MLflow guarda qual versão do modelo estava em produção, com quais
  parâmetros e sobre quais dados foi treinada;
- os limiares de decisão são configuração versionada, não número mágico
  enterrado no código.

**Necessário em produção, fora do escopo desta fase:**

- canal formal de contestação com **revisão humana** e prazo de resposta;
- explicação individual por decisão (SHAP ou equivalente), em linguagem
  acessível — o art. 20, §1º fala em informações claras e adequadas sobre os
  critérios;
- registro auditável de cada decisão emitida, com a versão do modelo, por todo
  o período de retenção;
- comunicação ao titular de que houve decisão automatizada.

## 5. Vieses e mitigação

### O que foi medido: idade

O modelo usa **idade** como feature, e ela carrega peso real (PSI de 0,549 no
lote `2024-04_novo_perfil`). Idade é critério protegido em várias jurisdições,
e no Brasil discriminação etária na oferta de crédito é tema sensível. Como o
atributo está no dado, o viés é **medido**, não só discutido:
`src/governance/fairness.py` (`make fairness`, e também dentro de
`make pipeline`) audita cada lote de produção em quatro faixas etárias e grava
`reports/equidade_idade.json`.

Três métricas, cada uma respondendo a uma pergunta diferente:

| Métrica | Pergunta | Limiar de alerta |
|---|---|---|
| **Razão de impacto desigual** (menor ÷ maior taxa de aprovação entre faixas) | a aprovação é parecida entre as faixas? | < 0,8 (regra dos quatro quintos) |
| **Diferença de igualdade de oportunidade** (maior − menor aprovação **entre bons pagadores**) | quem de fato paga tem a mesma chance de ser aprovado? | > 0,10 |
| **Inadimplência aprovada por faixa** | quando o modelo erra, quem paga a conta? | acompanhada no dashboard |

A regra dos quatro quintos é convenção de auditoria norte-americana (EEOC),
não limite legal brasileiro; usamos como limiar porque é a referência mais
difundida. O limiar de 10 p.p. é escolha deste projeto. Os dois estão em
`fairness.py` e no alerta `DisparidadeEtaria` do Prometheus — cruzou, dispara,
como os alertas de drift.

A linha de base é o lote `2024-01_estavel`: vem do pool futuro, nunca visto no
treino, e não tem drift. Medir na Referência mediria o modelo sobre dados que
ele memorizou.

### Resultado na linha de base

| Faixa | Participação | Inadimplência real | Aprovação | Aprovação de bons pagadores | Inadimplência aprovada |
|---|---|---|---|---|---|
| 18-29 | 5,6% | 7,4% | 60,6% | **64,2%** | 1,9% |
| 30-44 | 26,6% | 9,1% | 65,1% | 70,2% | 2,1% |
| 45-59 | 35,5% | 7,0% | 75,2% | 79,3% | 1,9% |
| 60+ | 32,4% | 3,2% | 91,1% | **92,8%** | 1,3% |

Razão de impacto desigual **0,665** e diferença de oportunidade **0,286**: os
dois alertas disparam já no mês sem drift. Parte da diferença de aprovação é
risco real (a faixa 60+ inadimple menos da metade), e é por isso que a métrica
que pesa é a segunda: **um bom pagador de 18 a 29 anos tem 64% de chance de ser
aprovado; um bom pagador acima de 60, 93%.** Note que a faixa 18-29 inadimple
*menos* que a 30-44 e ainda assim é a menos aprovada.

### O drift piorou a equidade?

| Lote | Razão de impacto | Dif. de oportunidade | Inadimpl. aprovada 18-29 | Inadimpl. aprovada 60+ |
|---|---|---|---|---|
| `2024-01_estavel` | 0,665 | 0,286 | 1,9% | 1,3% |
| `2024-02_inflacao_leve` | 0,679 | 0,247 | 2,1% | 1,5% |
| `2024-03_inflacao_forte` | 0,619 | 0,297 | 3,6% | 1,5% |
| `2024-04_novo_perfil` | 0,617 | **0,312** | 3,6% | 0,8% |
| `2024-05_recessao` | 0,622 | 0,224 | **25,7%** | 1,6% |

Duas leituras:

1. **No data drift, a disparidade de aprovação cresce.** Em `2024-04` a faixa
   18-29 passa de 6% para 25% da base, e a diferença de oportunidade chega ao
   pior valor (0,312). O modelo opera fora do domínio de treino justamente
   sobre o grupo que já tratava pior.
2. **No concept drift, o dano muda de lado.** Na recessão a diferença de
   oportunidade até *diminui* — mas porque o modelo continua aprovando jovens
   que agora inadimplem: **25,7% da carteira aprovada de 18 a 29 anos não
   paga**, contra 1,6% na faixa 60+. Para a fintech é prejuízo; para o
   cliente jovem é crédito concedido que ele não conseguiria pagar. Equidade
   não é só "aprovar igual": é também não empurrar dívida impagável para um
   grupo. Nenhuma métrica de aprovação sozinha enxerga isso, daí a terceira
   métrica estar no dashboard.

### Tirar a idade do modelo resolve?

Não. O mesmo estimador, retreinado na mesma Referência **sem** a coluna `age`:

| Lote | AUC com idade | AUC sem idade | Razão de impacto com → sem | Dif. de oportunidade com → sem |
|---|---|---|---|---|
| `2024-01_estavel` | 0,8711 | 0,8682 | 0,665 → 0,710 | 0,286 → 0,231 |
| `2024-04_novo_perfil` | 0,8254 | 0,8343 | 0,617 → 0,663 | 0,312 → 0,256 |
| `2024-05_recessao` | 0,8233 | 0,8118 | 0,622 → 0,685 | 0,224 → 0,147 |

A disparidade cai pouco e continua acima dos dois limiares em todos os lotes.
(De passagem: em `2024-04`, em que a própria idade sofreu drift, o modelo sem
idade tem AUC *maior* — uma feature que desloca com a demografia da base é
também uma fonte de fragilidade.)
O motivo é o **teste de proxy**: um classificador treinado para prever "tem de
18 a 29 anos" usando só as outras nove features acerta com **ROC AUC 0,918**.
Utilização de crédito, número de linhas abertas, financiamento imobiliário e
dependentes codificam a idade. Remover a coluna remove o rótulo, não a
informação — e ainda tira do auditor a forma mais direta de medir o problema.

### O que o dataset não permite verificar

O Give Me Some Credit **não traz** gênero, raça, região ou estado civil. Para
esses grupos este projeto **não consegue** medir disparidade, e dizer o
contrário seria falso. Ausência de atributo não é ausência de viés: o teste
de proxy acima mostra, com a idade, como as features carregam atributos que
não estão nominalmente no modelo — renda, endividamento e CEP fazem o mesmo
com raça e classe em qualquer carteira de crédito brasileira.

### Plano de mitigação

**Implementado neste projeto:**

- auditoria por faixa etária em todo lote, no mesmo pipeline que mede PSI;
- limiares de disparidade versionados em código e promovidos a alerta
  (`DisparidadeEtaria`) e a painel no Grafana;
- experimento "sem idade" e teste de proxy reproduzíveis com `make fairness`.

**Necessário em produção:**

1. **Decidir o uso de idade com o jurídico**, com os números acima na mesa: a
   idade vale 0,003 de AUC na linha de base e os proxies carregam a maior
   parte da disparidade. Retirá-la sozinha é cosmético; a decisão real é
   sobre o limiar e sobre restrições de equidade no treino.
2. **Mitigar no ponto certo**: limiar de decisão por faixa calibrado para
   igualar a aprovação de bons pagadores (pós-processamento), ou restrição de
   equidade no treino (in-processing). Ambos custam algum AUC — o trade-off
   tem de ser decidido com compliance, não pelo time de dados sozinho.
3. **Coletar atributos protegidos separadamente** (gênero, raça), com base
   legal própria e acesso restrito, usados **apenas** para auditoria — nunca
   como feature — e estender a mesma auditoria a eles.
4. **Revisão humana obrigatória** para recusas na faixa com alerta ativo,
   enquanto a disparidade não for tratada (ver art. 20, seção 4).

## 6. Retenção e eliminação (arts. 15 e 16)

| Dado | Retenção | Justificativa |
|---|---|---|
| Dataset de Referência (treino) | enquanto o modelo derivado estiver em produção + 5 anos | auditoria da decisão; art. 16, I (obrigação legal) |
| Lotes de produção com rótulo | 5 anos após o encerramento do contrato | prazo do Código Civil para pretensões; base para retreino |
| Score e decisão por proposta | 5 anos | direito de revisão do art. 20 e defesa em processo |
| Laudos de drift e métricas agregadas | 5 anos | **dado agregado, sem titular identificável** |
| Logs do pipeline | 12 meses | depuração operacional; não contêm dado de titular |
| Propostas recusadas | 2 anos | reject inference e contestação; prazo menor por não haver contrato |
| Modelos no MLflow Registry | indefinido (metadados) | linhagem; o artefato não contém dado pessoal |

**Eliminação.** Encerrado o prazo, os dados pessoais são eliminados ou
anonimizados de forma irreversível (art. 16). As métricas agregadas de drift
sobrevivem porque já são agregadas — ninguém é identificável num PSI.

**Requisição do titular (art. 18).** Pedidos de eliminação são atendidos nos
dados operacionais. Importante ser honesto: **um modelo já treinado não
"desaprende" um indivíduo**. A mitigação praticável é retreinar sem o titular
no próximo ciclo programado, e a política precisa declarar isso em vez de
prometer uma eliminação que a tecnologia não entrega.

## 7. Segurança (art. 46)

**Implementado neste projeto:**

- container roda como **usuário não-root**, build multi-stage sem ferramentas
  de build na imagem final;
- dependências travadas em `uv.lock`, com `--frozen` no build;
- **CodeQL** (`security-extended`) e **zizmor** no CI; actions fixadas por SHA;
- segredos fora do repositório (`.env` no `.gitignore`, `.env.example` versionado);
- o contrato de dados é, ele próprio, controle de segurança: barra injeção de
  coluna inesperada.

**Necessário em produção:**

criptografia em repouso e em trânsito, segregação de ambientes, controle de
acesso por papel ao dado bruto, trilha de auditoria de acesso, DPO designado,
RIPD formal e plano de resposta a incidentes com comunicação à ANPD no prazo
do art. 48.

## 8. Papéis

| Papel | Quem |
|---|---|
| **Controlador** | a fintech que concede o crédito |
| **Operador** | provedores de nuvem e serviços de dados contratados |
| **Encarregado (DPO)** | a designar; canal público de contato obrigatório (art. 41) |
| **Titular** | o solicitante do crédito |

## 9. Resumo executivo

Este projeto trata dados pessoais não sensíveis, sem identificadores diretos,
para análise de risco de crédito, com base legal no art. 7º, V e IX da LGPD.
A arquitetura aplica minimização por contrato (`strict=True` barra coluna não
declarada), mantém linhagem auditável de dado a decisão no MLflow e expõe os
critérios de decisão em toda resposta da API.

As lacunas conhecidas estão declaradas, não escondidas: não há revisão humana
formalizada, não há explicação individual por decisão, e o dataset público só
permite auditar equidade por idade — onde a auditoria encontrou disparidade
acima dos limiares em todos os lotes, que remover a idade não resolve. As três
são pré-requisitos para operação real e estão endereçadas no plano acima.
