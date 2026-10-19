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

### O que foi examinado

O modelo usa **idade** como feature, e a análise de drift mostrou que ela
carrega peso real (PSI de 0,549 no lote `2024-04_novo_perfil`). Idade é
critério protegido em várias jurisdições, e no Brasil discriminação etária na
oferta de crédito é tema sensível.

O cenário de concept drift documentado em
[`causal_analysis.md`](causal_analysis.md) é, no fundo, um achado de equidade:
o modelo **subestimava sistematicamente** o risco de jovens sem patrimônio
imobiliário. Num regime econômico ele os aprovava bem; noutro, o mesmo
comportamento virou prejuízo — e para o cliente, crédito concedido que ele não
conseguiria pagar.

### O que o dataset não permite verificar

O Give Me Some Credit **não traz** gênero, raça, região ou estado civil. Então
este projeto **não consegue** medir disparidade entre grupos protegidos —
e dizer o contrário seria falso. Ausência de atributo não é ausência de viés:
renda, endividamento e código postal funcionam como proxies de raça e classe
em qualquer carteira de crédito brasileira.

### Plano de mitigação para produção

1. **Coletar atributos protegidos separadamente**, com base legal própria e
   acesso restrito, usados **apenas** para auditoria de equidade — nunca como
   feature do modelo.
2. **Medir disparidade por grupo** a cada ciclo: taxa de aprovação, taxa de
   falso positivo e falso negativo, calibração por grupo. Acrescentar essas
   métricas ao mesmo pipeline de monitoramento que hoje mede PSI.
3. **Definir limiar de disparidade** com a área de compliance e tratá-lo como
   os alertas de drift: cruzou, dispara.
4. **Revisar o uso de idade** com jurídico — se o ganho preditivo justifica o
   risco regulatório, e se features correlacionadas conseguem substituí-la.
5. **Testar proxies**: treinar um classificador para prever o atributo
   protegido a partir das features do modelo. Se ele acerta bem, há proxy — e
   remover o atributo direto não resolve nada.

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
formalizada, não há explicação individual por decisão, e o dataset público não
permite auditoria de equidade entre grupos protegidos. As três são
pré-requisitos para operação real e estão endereçadas no plano acima.
