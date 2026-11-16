# Rubrica de Avaliação — Projeto 2

Critérios de correção de cada componente. As datas e os pesos são os do
[enunciado](ENUNCIADO.md).

## Como a nota é composta

| Componente                                         | Pontos  | Tipo       |
| -------------------------------------------------- | ------- | ---------- |
| 1. Submissão válida ao placar                      | 0,5     | grupo      |
| 2. Pacote reprodutível                             | 1,5     | grupo      |
| 3. Anexo de auditoria de IA                        | 0,5     | grupo      |
| 4. Relatório (ablação ou custo do estoque)         | 1,5     | grupo      |
| 5. Pôster e análise                                | 2,5     | grupo      |
| **Subtotal coletivo**                              | **6,5** |            |
| 6. Questionário individual (Canvas)                | 3,5     | individual |

```
nota final = (subtotal coletivo × multiplicador) + questionário
```

O multiplicador vem do questionário e vale entre 0,5 e 1,0. A tabela está
na seção "O multiplicador", logo depois da seção 6.

---

## 1. Submissão válida ao placar — 0,5

Binário, com um meio-termo.

| Situação                                                                                             | Pontos |
| ---------------------------------------------------------------------------------------------------- | ------ |
| Entregue no prazo, `verificar_submissao.py` passa sem `[ERRO]` e roda no placar sem desqualificação  | 0,5    |
| Roda no placar, mas falha em mais da metade das séries                                               | 0,25   |
| Desqualificada, ou entrega fora do prazo                                                             | 0      |

**Colocação não pontua.** Uma submissão que fica em último lugar recebe
0,5 desde que seja válida. Uma submissão que fica atrás das duas
referências recebe 0,5 também.

O que a submissão faz pela nota aparece nos componentes 4, 5 e 6.

---

## 2. Pacote reprodutível — 1,5

O critério é operacional: o que acontece quando o professor roda.

| Nível        | Pontos | Descritor                                                                                                                                  |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Insuficiente | 0      | Script não existe, ou não roda nem depois de ajuste                                                                                        |
| Básico       | 0,5    | Roda depois de intervenção manual (corrigir caminho, instalar algo não declarado, editar código)                                           |
| Proficiente  | 1,0    | Roda com um comando, regenera os números do pôster e do relatório, dados brutos completos e sem filtragem                                  |
| Avançado     | 1,5    | Além do acima: sementes fixadas em todo treino, versões declaradas, e o script falha com mensagem clara se algo estiver faltando            |

**Verificação de "sem filtragem":** o CSV precisa ter uma linha por
semente de cada configuração, incluindo as execuções que divergiram.
Reportar só a melhor semente, ou apagar a linha de uma configuração que
divergiu, derruba para Básico, mesmo que tudo rode.

**Verificação dos números:** escolha três números do pôster e confira
contra o que o script regenerou. Número que não bate, sem explicação no
README, derruba um nível.

---

## 3. Anexo de auditoria de IA — 0,5

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

## 4. Relatório — 1,5

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
| Básico       | 0,5    | Computação: menos de quatro componentes, ou uma semente por variante. Produção: só uma regra de estoque, ou só a razão de custos do enunciado                                    |
| Proficiente  | 1,0    | Computação: ao menos quatro componentes, três sementes ou mais, variação entre sementes reportada, conclusão sobre o que importa. Produção: as duas famílias, ao menos três razões de custo, recomendação em reais |
| Avançado     | 1,5    | Além do acima: Computação: o grupo testa uma interação (dois componentes que só importam juntos) ou mostra que um ganho é menor que a variação entre sementes. Produção: o grupo liga o quantil ótimo medido ao quantil crítico teórico e explica a diferença |

---

## 5. Pôster e análise — 2,5

### 5a. Correção técnica (0,8)

| Nível        | Pontos | Descritor                                                                                                                                                                                            |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Erro conceitual central (chama erro de treino de generalização, usa o teste do placar para escolher a configuração e reporta como validação, compara MSE de séries de volumes diferentes como se fossem comparáveis, chama de "aprendizado" uma regra sem treino) |
| Básico       | 0,25   | Sem erro central, mas afirmações imprecisas em pontos secundários                                                                                                                                    |
| Proficiente  | 0,55   | Terminologia correta, unidades declaradas, período de cada número declarado, comparações feitas entre coisas comparáveis                                                                             |
| Avançado     | 0,8    | Além do acima: o pôster distingue explicitamente o que foi medido do que foi inferido                                                                                                                |

### 5b. Qualidade da evidência (0,8)

| Nível        | Pontos | Descritor                                                                                                                                                   |
| ------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Números sem origem rastreável, ou gráfico sem eixo rotulado                                                                                                 |
| Básico       | 0,25   | Resultados de rede com uma semente só, sem menção a variabilidade                                                                                           |
| Proficiente  | 0,55   | Três sementes ou mais onde há aleatoriedade, execuções que divergiram reportadas, dados do placar incorporados                                              |
| Avançado     | 0,8    | Além do acima: o grupo identifica uma fonte de variação alheia ao modelo (semente, ordem dos minilotes, período avaliado) e mostra como a controlou          |

### 5c. Profundidade da interpretação (0,9)

| Nível        | Pontos | Descritor                                                                                                                                                                         |
| ------------ | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insuficiente | 0      | Tabelas sem leitura. O pôster mostra números e para                                                                                                                               |
| Básico       | 0,3    | Cada resultado acompanhado da explicação padrão de livro-texto                                                                                                                    |
| Proficiente  | 0,6    | Ao menos um resultado que contrariou a expectativa do grupo, explicado por uma característica específica desta série ou deste domínio, com o dado que sustenta a explicação apontado |
| Avançado     | 0,9    | Além do acima, uma das duas: um experimento adicional que discrimina entre duas causas possíveis, ou a comparação do desempenho da própria submissão na própria série com a média do placar, com a diferença explicada |

---

## 6. Questionário individual (Canvas) — 3,5

O Canvas sorteia três questões do banco público para cada aluno,
respondidas por escrito durante a aula de 01/12. O aluno pode consultar o
pôster e os dados do grupo.

### Pontuação por questão

Cada uma vale até 1,17, e a soma das três é limitada a 3,5. Escala
aplicada individualmente:

| Nível        | Fração | Descritor                                                                                                                       |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Não responde | 0      | Em branco, ou resposta que não trata da questão                                                                                 |
| Recita       | 0,25   | Reproduz a definição correta sem conectá-la ao trabalho do grupo                                                                |
| Aplica       | 0,6    | Conecta o conceito ao próprio trabalho, apontando o dado ou o trecho de código relevante                                        |
| Domina       | 1,0    | Além do acima: reconhece o limite da própria resposta, ou constrói um contraexemplo, ou diz que resultado esperaria de uma variação e por quê |

### Condições de avaliação

Avalia-se o conteúdo, não a redação. Uma resposta correta escrita de
forma simples vale o mesmo que uma resposta correta bem escrita.

Respostas idênticas, ou quase, entre alunos de grupos diferentes são
tratadas como cópia.

---

## O multiplicador

A nota do questionário também determina um multiplicador aplicado aos
6,5 pontos coletivos.

| Questionário             | Multiplicador |
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

| Aluno | Questionário | Multiplicador | Nota final       |
| ----- | ------------ | ------------- | ---------------- |
| A     | 3,2          | 1,00          | 5,8 + 3,2 = 9,0  |
| B     | 2,5          | 0,90          | 5,22 + 2,5 = 7,7 |
| C     | 1,0          | 0,60          | 3,48 + 1,0 = 4,5 |

---

## Prazos e penalidades

| Entrega                                  | Prazo            | Atraso                        |
| ---------------------------------------- | ---------------- | ----------------------------- |
| Submissão ao placar                      | 24/11, 23h59     | Não aceito. Componente zerado |
| Pacote reprodutível, relatório e pôster  | 29/11, 23h59     | Não aceito. Componente zerado |
| Questionário individual                  | 01/12, na aula   | Não aceito fora da aula       |

Nenhum prazo tem tolerância. O placar é uma execução coletiva única: não
há como rodá-lo de novo para quem perdeu a data.
