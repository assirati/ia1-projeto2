"""
Submissão ao placar do Projeto 2.

O professor treina a rede descrita em CONFIG com as features abaixo, em
cada uma das 30 séries do placar, e mede a previsão no período de teste
oculto (01/10 a 31/12/2025). Ver a seção 5 do ENUNCIADO.

Regras:
  - só a biblioteca padrão, e só os módulos math, statistics, datetime,
    itertools, functools, collections, bisect e heapq
  - não abra arquivos, não importe outros arquivos de vocês e não use
    getattr, .format nem atributos que começam com "_"
  - features devolve sempre o mesmo número de valores (no máximo 100)

Este exemplo é a referência do repositório: as features de
engine/features.basicas e a configuração padrão da rede. Substituam.
"""

NOME = "COLOQUE O NOME DO GRUPO AQUI"

# Configuração da rede (engine/rede.py). Chaves omitidas ficam no padrão.
CONFIG = {
    "camadas": [16],        # neurônios por camada oculta; [] = modelo linear
    "ativacao": "relu",     # identidade | sigmoide | tanh | relu | leaky_relu
    "perda": "mse",         # mse | mae | huber | pinball
    "quantil": 0.5,         # só com perda="pinball"
    "otimizador": "adam",   # sgd | momento | adam
    "taxa": 0.01,
    "epocas": 200,          # até 500
    "lote": 32,
    "l2": 0.0,
    "inicializacao": "he",  # xavier | he | normal | zeros
    "normalizar": True,
}


def features(hist, dia):
    """hist: dias anteriores, com "demanda" (hist[-1] é ontem).
    dia: o dia a prever, sem "demanda". Devolve uma lista de números."""
    semana = [1.0 if dia["dia_semana"] == k else 0.0 for k in range(7)]
    d = [l["demanda"] for l in hist[-28:]]
    return semana + [
        dia["feriado"], dia["vespera_feriado"], dia["evento"],
        dia["promocao"], dia["desconto"], dia["temperatura"],
        d[-1], d[-7], sum(d[-7:]) / 7, sum(d) / len(d),
    ]


# Opcional. Quantas unidades estocar para o dia, dada a previsão da rede.
# Só afeta as colunas de custo do estoque. Sem esta função, estoca-se a
# própria previsão.
#
# def decidir(previsao, hist, dia):
#     return previsao
