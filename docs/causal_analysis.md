# Análise Causal da Degradação

> "O AUC caiu 0,045" não é diagnóstico — é sintoma.

O diagnóstico responde **por qual mecanismo** caiu, porque a resposta
operacional é diferente em cada caso:

| Mecanismo | O que aconteceu | Resposta correta | Resposta errada |
|---|---|---|---|
| **Data drift** | `P(X)` mudou; `P(Y\|X)` continua valendo | recalibrar limiar, reponderar amostra | retreinar às cegas |
| **Concept drift** | `P(Y\|X)` mudou — o aprendido virou mentira | **retreinar** com dado do regime novo | recalibrar (não resolve) |

Tratar concept drift com recalibração é como ajustar o termômetro porque o
paciente está com febre.

## O modelo causal

```
                    ┌──────────────────────────┐
                    │  Choque macroeconômico   │  ← não observado pelo modelo
                    │  (inflação + recessão)   │
                    └────┬────────────────┬────┘
                         │                │
           ┌─────────────┘                └──────────────┐
           ▼                                             ▼
   ┌───────────────────┐                      ┌──────────────────────┐
   │ Features medidas  │                      │ Propensão real a     │
   │ • renda nominal ↑ │                      │ inadimplir, dado o   │
   │ • endividamento ↑ │                      │ mesmo perfil ↑       │
   │ • idade da base ↓ │                      │ (jovem sem colateral)│
   └─────────┬─────────┘                      └──────────┬───────────┘
             │                                           │
             │  caminho A: DATA DRIFT                    │  caminho B: CONCEPT DRIFT
             │  visível em PSI/KS                        │  invisível em PSI/KS
             │                                           │
             └──────────────┬────────────────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Decisão errada de    │
                 │ crédito → prejuízo   │
                 └──────────────────────┘
```

O choque macroeconômico é a **causa comum** dos dois caminhos — e é
exatamente a variável que o modelo não observa. Ele vê renda, idade e
endividamento; não vê "estamos numa recessão". Por isso um modelo treinado num
regime econômico tem validade limitada a esse regime: não é defeito do
algoritmo, é limite do que foi medido.

O caminho B é o perigoso porque é **invisível na entrada**. As features chegam
com a mesma distribuição de sempre; o que mudou foi o mundo do outro lado
delas.

## Experimento de atribuição

Separar os dois caminhos em produção costuma ser difícil, porque só se observa
o efeito combinado. Aqui é possível fazer de forma limpa: nós controlamos o
processo gerador em `src/drift/simulate.py`, então dá para rodar o desenho
fatorial 2×2 — aplicar cada mecanismo **isolado** sobre a mesma amostra-base,
com a mesma semente.

É o desenho contrafactual clássico: mesma amostra, mesmo modelo, só muda o
mecanismo ligado.

```bash
make attribution    # ou: uv run python -m src.drift.attribution
```

### Resultados

Baseline de validação: ROC AUC 0,8687 · inadimplência da carteira aprovada 2,00%.

| Condição | `P(X)` | `P(Y\|X)` | ROC AUC | Δ AUC | Inadimplência | Carteira aprovada ruim | PSI máx | nº em drift |
|---|---|---|---|---|---|---|---|---|
| controle | — | — | 0,8691 | **+0,0005** | 6,69% | 1,94% | 0,002 | 0 |
| só data drift | muda | intacta | 0,8398 | −0,0289 | 6,69% | 1,66% | **0,561** | **3** |
| só concept drift | intacta | muda | 0,8193 | −0,0494 | 10,47% | 4,66% | **0,002** | **0** |
| ambos | muda | muda | 0,7196 | **−0,1490** | 17,46% | **10,02%** | 0,561 | 3 |

Duas validações de que o desenho está correto:

- o **controle** move o AUC em +0,0005 — ruído amostral desprezível;
- em **só data drift**, a taxa de inadimplência fica idêntica ao controle
  (6,69%), confirmando que `P(Y|X)` foi preservada. Era exatamente o que se
  queria isolar.

### Decomposição

```json
{
  "ruido_amostral":       0.0005,
  "efeito_data_drift":   -0.0294,
  "efeito_concept_drift": -0.0499,
  "efeito_conjunto":     -0.1495,
  "interacao":           -0.0702,
  "share_data_drift":     0.3707,
  "share_concept_drift":  0.6293
}
```

## Três conclusões

### 1. O mecanismo dominante é o invisível

O concept drift responde por **63%** do dano isolado — e é justamente o que
produz PSI de 0,002 e zero features em alerta. Um monitoramento que olhasse só
distribuição de features capturaria 37% do problema e deixaria passar a maior
parte, em silêncio.

É a justificativa quantitativa para a arquitetura do projeto: **drift de dados
e desempenho do modelo precisam ser monitorados juntos.** O alerta
`degradacao_silenciosa` só existe porque as duas fontes são cruzadas.

### 2. Os mecanismos se potencializam

O efeito conjunto (−0,1495) é muito pior que a soma dos isolados (−0,0793). A
**interação vale −0,0702** — sozinha, maior que qualquer um dos efeitos
isolados.

A leitura: quando a base de clientes já se deslocou para fora do domínio de
treino, o modelo fica *mais* vulnerável a uma mudança de relação. Ele está
extrapolando em duas dimensões ao mesmo tempo. Consequência prática — não dá
para estimar o risco de uma recessão somando os efeitos observados
isoladamente; o cenário combinado precisa ser testado como cenário.

### 3. O custo real não está no AUC

A métrica que a diretoria entende: no cenário combinado, a inadimplência
**dentro da carteira que o modelo aprovou** vai de 1,94% para **10,02%** —
5,2× o prejuízo — com a taxa de aprovação praticamente intacta. O modelo
continua aprovando a mesma fatia de gente; é a qualidade dessa fatia que
desmoronou.

É por isso que `approved_bad_rate` é métrica de primeira classe no dashboard,
e não apenas AUC.

## O que fazer com o diagnóstico

| Sinal | Diagnóstico | Ação |
|---|---|---|
| PSI alto, desempenho estável | data drift benigno | monitorar; checar se há extrapolação |
| PSI alto, desempenho caindo | data drift com impacto | recalibrar limiar; avaliar reponderação |
| **PSI baixo, desempenho caindo** | **concept drift** | **retreinar — recalibrar não resolve** |
| Contrato violado | falha de qualidade a montante | corrigir a origem; o lote já foi bloqueado |

Uma ressalva honesta: a atribuição limpa deste documento é possível porque
**nós** geramos o drift e conhecemos a verdade. Em produção, o rótulo chega com
atraso (no crédito, 1 a 2 anos para confirmar inadimplência), e esse atraso é o
obstáculo real. Mitigações usuais — e fora do escopo desta fase: rótulos
substitutos de curto prazo (atraso de 30/60 dias), *reject inference* para o
viés de quem foi recusado, e monitoramento de estabilidade do score enquanto o
rótulo definitivo não chega. O que este projeto entrega é a camada que torna o
diagnóstico possível **assim que** o rótulo chegar.
