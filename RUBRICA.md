# Rubrica de Avaliação — Projeto 2

Critérios de correção de cada componente. As datas e os pesos são os do
[enunciado](ENUNCIADO.md).

## Como a nota é composta

| Componente                                         | Pontos  | Tipo       |
| -------------------------------------------------- | ------- | ---------- |
| 1. Pré-registro de hipóteses                       | 1,5     | grupo      |
| 2. Submissão válida ao placar                      | 0,5     | grupo      |
| 3. Pacote reprodutível                             | 1,0     | grupo      |
| 4. Anexo de auditoria de IA                        | 0,5     | grupo      |
| 5. Relatório (ablação ou custo do estoque)         | 1,0     | grupo      |
| 6. Pôster e análise                                | 1,5     | grupo      |
| 7. Ficha de replicação ao vivo                     | 0,5     | grupo      |
| **Subtotal coletivo**                              | **6,5** |            |
| 8. Arguição individual                             | 3,5     | individual |

```
nota final = (subtotal coletivo × multiplicador) + arguição
```

O multiplicador vem da arguição e vale entre 0,5 e 1,0. A tabela está na
seção "O multiplicador", logo depois da seção 8.

---

## 1. Pré-registro de hipóteses — 1,5

Avaliado em 19/11, com nota lançada só depois do pôster. A qualidade da
predição e a qualidade da análise da divergência são avaliadas juntas.

### 1a. Falsificabilidade das predições (0,75)

| Nível        | Pontos | Descritor                                                                                                                                         |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Predições sem número. "A rede será melhor", "MAE vai ajudar"                                                                                      |
| Básico       | 0,25   | Predições com número em menos da metade dos itens obrigatórios, ou faixas tão largas que qualquer resultado cabe (ex.: "WAPE entre 5% e 60%")     |
| Proficiente  | 0,5    | Todos os itens obrigatórios com número ou faixa fechada. Justificativas ligam a predição a um mecanismo do modelo, não a intuição                 |
| Avançado     | 0,75   | Além do acima, ao menos uma predição contraria o resultado esperado do livro-texto e a justificativa explica por que esta série seria diferente   |

**Regra explícita:** predição vaga que acerta pontua **menos** que predição
arriscada que erra. Um grupo que escreveu "a rede terá erro menor que o
ingênuo" e acertou fica no nível Insuficiente deste critério.

### 1b. Análise da divergência, verificada no pôster (0,75)

| Nível        | Pontos | Descritor                                                                                                     |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | O pôster não menciona o pré-registro, ou trata divergências como erro de medição sem investigar               |
| Básico       | 0,25   | Divergências reportadas, com explicação genérica ("a série era diferente", "faltou treinar mais")             |
| Proficiente  | 0,5    | Ao menos uma divergência explicada por um mecanismo específico, com o dado que sustenta a explicação apontado |
| Avançado     | 0,75   | Explicação acompanhada de um experimento adicional que discrimina entre duas causas possíveis                 |

---

## 2. Submissão válida ao placar — 0,5

Binário, com um meio-termo.

| Situação                                                                                             | Pontos |
| ---------------------------------------------------------------------------------------------------- | ------ |
| Entregue no prazo, `verificar_submissao.py` passa sem `[ERRO]` e roda no placar sem desqualificação  | 0,5    |
| Roda no placar, mas falha em mais da metade das séries                                               | 0,25   |
| Desqualificada, ou entrega fora do prazo                                                             | 0      |

**Colocação não pontua.** Uma submissão que fica em último lugar recebe
0,5 desde que seja válida. Uma submissão que fica atrás das duas
referências recebe 0,5 também.

O que a submissão faz pela nota aparece nos componentes 5, 6 e 8.

---

## 3. Pacote reprodutível — 1,0

O critério é operacional: o que acontece quando outro grupo roda.

| Nível        | Pontos | Descritor                                                                                                                                  |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Insuficiente | 0      | Script não existe, ou não roda nem depois de ajuste                                                                                        |
| Básico       | 0,3    | Roda depois de intervenção manual (corrigir caminho, instalar algo não declarado, editar código)                                           |
| Proficiente  | 0,7    | Roda com um comando, regenera os números do pôster e do relatório, dados brutos completos e sem filtragem                                  |
| Avançado     | 1,0    | Além do acima: sementes fixadas em todo treino, versões declaradas, e o script falha com mensagem clara se algo estiver faltando            |

**Verificação de "sem filtragem":** o CSV precisa ter uma linha por
semente de cada configuração, incluindo as execuções que divergiram.
Reportar só a melhor semente, ou apagar a linha de uma configuração que
divergiu, derruba para Básico, mesmo que tudo rode.

---

## 4. Anexo de auditoria de IA — 0,5

| Nível        | Pontos | Descritor                                                                                                                             |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Anexo ausente, ou afirma que a IA não errou                                                                                           |
| Básico       | 0,25   | Erro relatado, mas trivial (erro de sintaxe, nome de variável) ou detectado pelo próprio interpretador                                |
| Proficiente  | 0,4    | Erro substantivo, com a descrição de como foi detectado                                                                               |
| Avançado     | 0,5    | Além do acima: o grupo aponta o que teria acontecido se o erro passasse, e descreve a verificação que passaram a fazer por causa dele |

**O que conta como substantivo:** derivada da ativação ou da perda escrita
errado (sinal da pinball, derivada da sigmoide), afirmação de que a rede
sempre vence o modelo linear, confusão entre o que MSE e MAE estimam,
feature que olha o futuro (média móvel centrada) apresentada como válida,
métrica calculada sobre o período errado, número plausível que não bateu
ao rodar, explicação confiante e incorreta sobre o comportamento do
treino.

---

## 5. Relatório — 1,0

No máximo três páginas, entregue junto com o pacote. O conteúdo muda com
o curso.

**Computação: relatório de ablação.** Partindo da submissão final, o grupo
retira (ou troca pelo padrão) um componente por vez, entre features e
itens da CONFIG, e mede o efeito na validação.

**Produção: análise de custo do estoque.** O grupo mede como o custo varia
com a regra de estoque (quantil da pinball, ou previsão MSE mais estoque
de segurança), compara as duas famílias de regra e mede a sensibilidade à
razão entre custo de falta e custo de sobra.

| Nível        | Pontos | Descritor                                                                                                                                                                       |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Relatório ausente, ou uma tabela sem leitura                                                                                                                                    |
| Básico       | 0,3    | Computação: menos de quatro componentes, ou uma semente por variante. Produção: só uma regra de estoque, ou só a razão de custos do enunciado                                    |
| Proficiente  | 0,7    | Computação: ao menos quatro componentes, três sementes ou mais, variação entre sementes reportada, conclusão sobre o que importa. Produção: as duas famílias, ao menos três razões de custo, recomendação em reais |
| Avançado     | 1,0    | Além do acima: Computação: o grupo testa uma interação (dois componentes que só importam juntos) ou mostra que um ganho é menor que a variação entre sementes. Produção: o grupo liga o quantil ótimo medido ao quantil crítico teórico e explica a diferença |

---

## 6. Pôster e análise — 1,5

### 6a. Correção técnica (0,5)

| Nível        | Pontos | Descritor                                                                                                                                                                                            |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Erro conceitual central (chama erro de treino de generalização, usa o teste do placar para escolher a configuração e reporta como validação, confunde WAPE com acurácia, chama de "aprendizado" uma regra sem treino) |
| Básico       | 0,15   | Sem erro central, mas afirmações imprecisas em pontos secundários                                                                                                                                    |
| Proficiente  | 0,35   | Terminologia correta, unidades declaradas, período de cada número declarado, comparações feitas entre coisas comparáveis                                                                             |
| Avançado     | 0,5    | Além do acima: o pôster distingue explicitamente o que foi medido do que foi inferido                                                                                                                |

### 6b. Qualidade da evidência (0,5)

| Nível        | Pontos | Descritor                                                                                                                                                   |
| ------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Números sem origem rastreável, ou gráfico sem eixo rotulado                                                                                                 |
| Básico       | 0,15   | Resultados de rede com uma semente só, sem menção a variabilidade                                                                                           |
| Proficiente  | 0,35   | Três sementes ou mais onde há aleatoriedade, execuções que divergiram reportadas, dados do placar incorporados                                              |
| Avançado     | 0,5    | Além do acima: o grupo identifica uma fonte de variação alheia ao modelo (semente, ordem dos minilotes, período avaliado) e mostra como a controlou          |

### 6c. Profundidade da interpretação (0,5)

| Nível        | Pontos | Descritor                                                                                                                              |
| ------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Tabelas sem leitura. O pôster mostra números e para                                                                                    |
| Básico       | 0,15   | Cada resultado acompanhado da explicação padrão de livro-texto                                                                         |
| Proficiente  | 0,35   | Ao menos um resultado explicado por uma característica específica desta série ou deste domínio                                         |
| Avançado     | 0,5    | O grupo compara o desempenho da própria submissão na própria série com o desempenho na média do placar, e explica a diferença          |

---

## 7. Ficha de replicação ao vivo — 0,5

Preenchida em 01/12 pelo grupo auditor.

| Nível        | Pontos | Descritor                                                                                                              |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Ficha sem evidência de que o script foi rodado (campos de valor regenerado vazios ou idênticos ao pôster sem execução) |
| Básico       | 0,25   | Ficha completa, três números conferidos, nada além disso                                                               |
| Proficiente  | 0,4    | Além do acima: a pergunta do campo 4 é sobre algo que só se percebe rodando o código                                   |
| Avançado     | 0,5    | O auditor encontra uma divergência real, ou uma afirmação não sustentada pelos dados brutos, e a descreve com precisão |

**Penalidade cruzada.** Se o professor detectar depois um erro grave que o
auditor deixou passar, ambos os grupos perdem 0,25: o auditor neste
componente, o auditado no componente 3.

---

## 8. Arguição individual — 3,5

Três perguntas, 6 a 8 minutos. Duas sorteadas do banco público, uma
improvisada a partir do pôster e dos dados do grupo.

### Pontuação por pergunta

Cada uma vale até 1,17, e a soma das três é limitada a 3,5. Escala
aplicada individualmente:

| Nível        | Fração | Descritor                                                                                                                                               |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Não responde | 0      | Silêncio, ou resposta que não trata da pergunta                                                                                                         |
| Recita       | 0,25   | Reproduz a definição correta sem conectá-la ao trabalho do grupo                                                                                        |
| Aplica       | 0,6    | Conecta o conceito ao próprio trabalho, apontando o dado ou o trecho de código relevante                                                                |
| Domina       | 1,0    | Além do acima: reconhece o limite da própria resposta, ou constrói um contraexemplo, ou prevê corretamente o resultado de uma variação proposta na hora |

### Sobre a terceira pergunta

Ela é o discriminante do projeto. Formatos:

> "Sua submissão ficou em 2º na sua série e em 6º na média do placar. O
> que isso diz?"

> "Troca a perda para MAE e roda. O que você espera antes de apertar
> enter?"

> "Por que a sua série teve WAPE de 14% e a do grupo ao lado de 21%, com
> a mesma configuração?"

Quem produziu os dados responde. Quem recebeu os resultados prontos, não.

### Condições de avaliação

Respostas podem ser dadas apontando para o pôster, escrevendo no papel ou
rodando código na hora. Avalia-se o conteúdo, não a desenvoltura.

Hesitação, gagueira e nervosismo não descontam. Uma resposta correta dita
mal vale o mesmo que uma resposta correta dita bem.

---

## O multiplicador

A nota da arguição também determina um multiplicador aplicado aos 6,5
pontos coletivos.

| Arguição                 | Multiplicador |
| ------------------------ | ------------- |
| 3,0 ou mais              | 1,00          |
| de 2,3 a menos de 3,0    | 0,90          |
| de 1,5 a menos de 2,3    | 0,75          |
| de 0,8 a menos de 1,5    | 0,60          |
| abaixo de 0,8            | 0,50          |

**Por que isso existe.** Quem não consegue explicar o próprio trabalho não
carrega a nota do grupo. O mecanismo resolve o problema do carona sem
exigir que ninguém denuncie ninguém.

**Exemplo.** Grupo com 5,8 de subtotal coletivo:

| Aluno | Arguição | Multiplicador | Nota final       |
| ----- | -------- | ------------- | ---------------- |
| A     | 3,2      | 1,00          | 5,8 + 3,2 = 9,0  |
| B     | 2,5      | 0,90          | 5,22 + 2,5 = 7,7 |
| C     | 1,0      | 0,60          | 3,48 + 1,0 = 4,5 |

---

## Prazos e penalidades

| Entrega                                  | Prazo            | Atraso                        |
| ---------------------------------------- | ---------------- | ----------------------------- |
| Pré-registro                             | 19/11, 23h59     | Não aceito. Componente zerado |
| Submissão ao placar                      | 24/11, 23h59     | Não aceito. Componente zerado |
| Pacote reprodutível, relatório e pôster  | 29/11, 23h59     | Não aceito. Componente zerado |
| Ficha de replicação                      | 01/12, na sessão | Não aceito fora da sessão     |

Nenhum prazo tem tolerância. O pré-registro depende de ser anterior à
execução, e o placar é uma execução coletiva única: não há como rodá-lo
de novo para quem perdeu a data.
