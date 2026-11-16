# Projeto 2 — Inteligência Artificial 1

Aprendizado de máquina e redes neurais aplicados à previsão de demanda de
um centro de distribuição.

Leia o [enunciado](ENUNCIADO.md) para as regras completas. Este arquivo é só para você
colocar o projeto para rodar em cinco minutos.

---

## Começar

Requer Python 3.10 ou superior. Em macOS e Linux, troque `python` por
`python3` em todos os comandos deste arquivo, se `python` não existir.

Clonem o repositório e instalem as dependências:

```bash
git clone https://github.com/assirati/ia1-projeto2.git
cd ia1-projeto2
python -m pip install -r requirements.txt
```

O `requirements.txt` instala `numpy`, que a engine usa (a rede neural é
escrita em numpy), e `pandas` e `matplotlib`, que vocês vão usar para
análise e gráficos.

Se o pip recusar a instalação com o erro `externally-managed-environment`
(comum em Linux e no Python do Homebrew), criem um ambiente virtual e
instalem dentro dele:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Depois, vejam a série do grupo:

```bash
python minha_serie.py <matricula> <producao|computacao>
```

Os grupos são os mesmos do Projeto 1. Usem a mesma matrícula do P1 (a
do integrante mais velho do grupo). A mesma série vale para todas as
entregas.

Exemplo:

```bash
python minha_serie.py 20231234 computacao
```

Isso imprime o número da série, o arquivo CSV dela em `series/`, a
capacidade da doca normal e um resumo da demanda. A seção 2 do
[enunciado](ENUNCIADO.md) explica como ler essa saída.

## Primeiros resultados

```bash
python exemplos/exemplo_modulo1.py <matricula> <curso>
python exemplos/exemplo_modulo2.py <matricula> <curso>
```

Cada um roda os modelos de referência e grava um CSV em `dados/`.

**Isso não é a entrega.** É o esqueleto para vocês adaptarem. O trabalho
que vale nota começa quando vocês perguntam por que os números saíram
assim.

O segundo script demora cerca de um minuto, porque treina cada rede com
três sementes.

## Sua submissão ao placar

Editem `submissao/previsor.py`. Ele tem três partes:

| Parte                      | O que é                                                  |
| -------------------------- | -------------------------------------------------------- |
| `CONFIG`                   | a configuração da rede de `engine/rede.py`               |
| `features(hist, dia)`      | as entradas da rede, calculadas a partir do passado      |
| `decidir(previsao, hist, dia)` | opcional: quanto estocar, dada a previsão            |

O arquivo já vem com um exemplo funcional dentro (a referência do
placar). Substituam.

Antes de enviar:

```bash
python verificar_submissao.py <matricula> <curso>
```

Esse script roda as mesmas checagens do placar, na sua série e em mais
duas do pool, treinando no treino e medindo na validação. Ele diz se a
submissão é **válida**, não se ela é boa.

Marcações:

- `[ok]` passou
- `[!]` aviso. Não impede o envio, mas você precisa saber explicar
- `[ERRO]` submissão inválida. Corrija antes de 24/11

### Entrega no Canvas

Quando o verificador disser "Submissão válida", gerem um zip com o
arquivo. Usem no nome a mesma matrícula da série:

```bash
python -m zipfile -c submissao_<matricula>.zip submissao/previsor.py
```

O comando usa o próprio Python, então funciona igual em Windows, macOS e
Linux. Confiram o conteúdo antes de enviar:

```bash
python -m zipfile -l submissao_<matricula>.zip
```

A lista deve ter exatamente `previsor.py`, sem pasta em volta e sem
outros arquivos. O placar carrega o arquivo sozinho, então ele não pode
importar outros arquivos de vocês, nem a `engine`.

Enviem o zip na tarefa do placar no Canvas. Prazo: **24/11, 23h59**. Sem
segunda chamada.

## Documentos para preencher

O modelo de pôster fica no Google Drive (o link também está no Canvas).
O link abaixo abre a opção de fazer uma cópia na conta Google de vocês.
Façam a cópia, preencham e enviem na tarefa do pôster no Canvas:

| Documento | Quando | Onde enviar |
|---|---|---|
| [Modelo de pôster (A1)](https://docs.google.com/presentation/d/PREENCHER_POSTER_P2/copy) | até **29/11, 23h59**, junto com o pacote e o relatório | tarefa do pôster no Canvas |

O questionário individual de 01/12 é feito no próprio Canvas, durante a
aula, com questões sorteadas do [banco público](BANCO_PERGUNTAS.md).

---

## O que tem em cada lugar

```
ENUNCIADO.md              regras, prazos, pontuação
RUBRICA.md                critérios de correção de cada componente
BANCO_PERGUNTAS.md        as 24 questões do questionário individual
histograma_calibracao.png distribuição de dificuldade das séries

minha_serie.py            mostra a série do grupo
verificar_submissao.py    autoteste antes de enviar

series/                   as séries do pool (treino e validação)
submissao/previsor.py     o arquivo que vocês entregam
exemplos/                 scripts de partida, geram CSV
dados/                    saída dos scripts (vazio no início)

engine/
  config.py               custos, limites e atribuição de séries
  dados.py                leitura das séries, montagem de X e y, divisões
  features.py             funções de features de referência
  modelos.py              referências, regressão linear, k-NN, logística, perceptron
  rede.py                 a rede neural: backpropagation, otimizadores, instrumentação
  metricas.py             MSE, RMSE, viés, custo do estoque, classificação
  avaliacao.py            as regras do placar (o verificador usa este código)
```

## Referência rápida da API

**Dados e features**

```python
from engine.config import atribuir
from engine.dados import carregar_serie, montar, dividir
from engine import features as F

seed = atribuir("20231234", "computacao")
s = carregar_serie(seed, "computacao")
s.capacidade                                  # capacidade da doca normal
s.linhas[t]["demanda"], s.partes[t]           # dia t e a parte dele

X, y, idx = montar(s, F.basicas, "treino")                  # regressão
X, y, idx = montar(s, F.basicas, "treino", alvo="doca_extra")  # classificação
Xa, ya, Xb, yb = dividir(X, y, modo="aleatorio", frac_val=0.2, seed=0)
```

**Modelos**

```python
from engine.modelos import (RegressaoLinear, KNN, RegressaoLogistica,
                            Perceptron, prever_regra, sazonal)

RegressaoLinear(l2=1.0).treinar(X, y).prever(Xv)
KNN(k=5).treinar(X, y).prever(Xv)
RegressaoLogistica().treinar(X, c).prever_classe(Xv, limiar=0.5)
Perceptron(epocas=50).treinar(X, c).historico["erros"]
p, y, idx = prever_regra(s, sazonal, "validacao")
```

**Rede neural**

```python
from engine.rede import MLP, checar_gradiente

rede = MLP(camadas=[16], ativacao="relu", perda="mse", taxa=0.01,
           epocas=200, seed=0).treinar(X, y, Xv, yv)
rede.prever(Xv)
rede.historico     # perda_treino, perda_val (por época), divergiu
checar_gradiente(MLP(camadas=[8]), X[:64], y[:64])
```

**Métricas**

```python
from engine.metricas import mse, regressao, custo_estoque, classificacao, custo_doca

regressao(y, p)          # mse, rmse, rmse_rel (RMSE / demanda média, %), vies
custo_estoque(y, q)      # custo_100, atendimento, dias_com_falta, ...
classificacao(y, c)      # acuracia, precisao, revocacao, f1, matriz
custo_doca(y, c)         # custo da decisão de abrir a doca extra
```

## Perguntas frequentes

**A série de outro grupo é mais fácil que a minha?** \
As séries foram filtradas para uma banda de dificuldade. A distribuição
está em `histograma_calibracao.png`. Diferenças de erro entre séries são
esperadas e não mudam nenhuma conclusão qualitativa. Para comparar séries
de volumes diferentes, usem o RMSE relativo, não o MSE.

**Posso usar outras séries do pool?** \
Sim. Todas as séries de `series/` estão lá de propósito: o placar mede a
submissão em 30 delas, não só na sua.

**Posso usar numpy, pandas, scikit-learn, PyTorch?** \
Para análise e gráficos, sim. Para treinar a submissão, não: o placar usa
a rede de `engine/rede.py`, com a sua `CONFIG`. No arquivo
`submissao/previsor.py`, só a biblioteca padrão, com os módulos listados
na seção 5 do enunciado.

**Posso modificar o `engine/`?** \
Para experimentar, à vontade. Mas o placar e a conferência do pacote
rodam com a versão original, então não dependam das suas mudanças.

**Posso usar IA generativa?** \
Sim, sem restrição. Leia a seção 9 do enunciado: o pacote precisa
documentar um erro que a IA cometeu e como vocês detectaram.

**Por que a rede dá resultados diferentes a cada semente?** \
A inicialização dos pesos e a ordem dos minilotes são sorteadas. Isso não
é um erro: é uma fonte de variação que vocês precisam medir e reportar.
