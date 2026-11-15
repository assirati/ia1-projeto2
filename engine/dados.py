"""
Leitura das séries e montagem das matrizes de treino.

Cada linha da série é um dia:

    data             'AAAA-MM-DD'
    dia_semana       0 = segunda ... 6 = domingo
    feriado          1 se feriado nacional
    vespera_feriado  1 se o dia seguinte é feriado
    evento           1 em dias de campanha do varejo (Black Friday, semana antes do Natal)
    promocao         1 se o produto está em promoção
    desconto         fração de desconto (0,10 = 10%)
    preco            preço praticado no dia, já com o desconto (R$)
    temperatura      temperatura média prevista para o dia (°C)
    demanda          unidades pedidas ao centro de distribuição

Tudo menos `demanda` é conhecido na véspera. É isso que permite prever.

Uma função de features recebe `(hist, dia)`:

    hist   os dias ANTERIORES, do mais antigo ao mais recente, com `demanda`.
           hist[-1] é ontem, hist[-7] é o mesmo dia da semana passada.
    dia    o dia a prever, sem a chave `demanda`.

e devolve uma lista de números, sempre do mesmo tamanho. Ela não consegue
ver o futuro: `hist` termina na véspera. Por isso não existe vazamento de
dados do futuro neste projeto, desde que as features venham só de `hist`
e `dia`.
"""

import csv
import math
from collections.abc import Sequence
from types import MappingProxyType

import numpy as np

from .config import AQUECIMENTO, caminho_serie

_INTEIROS = ("dia_semana", "feriado", "vespera_feriado", "evento", "promocao")
_REAIS = ("desconto", "preco", "temperatura")


class Historico(Sequence):
    """Visão somente leitura dos primeiros `n` dias da série."""

    __slots__ = ("_linhas", "_n")

    def __init__(self, linhas, n):
        self._linhas = linhas
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._linhas[k] for k in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("o histórico só vai até a véspera")
        return self._linhas[i]

    def __iter__(self):
        for k in range(self._n):
            yield self._linhas[k]

    def __repr__(self):
        return f"Historico({self._n} dias)"


class Serie:
    """Uma série de demanda.

    serie.linhas[t]          dia t (somente leitura), com `demanda`
    serie.partes[t]          'treino', 'validacao' ou 'teste'
    serie.indices('treino')  índices dos dias daquela parte
    serie.capacidade         capacidade de recebimento da doca normal
                             (unidades/dia). Acima disso é preciso abrir a
                             doca extra. É o percentil 85 da demanda de treino.
    """

    def __init__(self, seed, curso, registros):
        self.seed = seed
        self.curso = curso
        self.partes = [r["parte"] for r in registros]
        linhas = []
        dias = []
        self._brutas = []  # os dicts por trás de cada linha (só a engine mexe)
        for r in registros:
            d = {"data": r["data"]}
            for k in _INTEIROS:
                d[k] = int(r[k])
            for k in _REAIS:
                d[k] = float(r[k])
            dem = r.get("demanda")
            d["demanda"] = None if dem in (None, "") else int(dem)
            dia = dict(d)
            del dia["demanda"]
            self._brutas.append(d)
            linhas.append(MappingProxyType(d))
            dias.append(MappingProxyType(dia))
        self.linhas = linhas
        self._dias = dias
        trein = sorted(l["demanda"] for l, p in zip(linhas, self.partes)
                       if p == "treino")
        self.capacidade = int(trein[int(0.85 * (len(trein) - 1))])

    def __len__(self):
        return len(self.linhas)

    def __repr__(self):
        return f"Serie(seed={self.seed}, curso={self.curso!r}, {len(self)} dias)"

    def indices(self, *partes):
        """Índices dos dias das partes pedidas, sem o aquecimento."""
        return [t for t, p in enumerate(self.partes)
                if p in partes and t >= AQUECIMENTO]

    def hist(self, t):
        """O que uma função de features enxerga ao prever o dia t."""
        return Historico(self.linhas, t)

    def dia(self, t):
        return self._dias[t]

    def trocar_futuro(self, t):
        """Troca, NESTA série, a demanda do dia t em diante por outro valor e
        devolve uma função que desfaz a troca. O placar usa isto para
        conferir que as features do dia t não dependem de nada a partir de
        t, por qualquer caminho (ver engine/avaliacao.py)."""
        originais = [(d, d["demanda"]) for d in self._brutas[t:]]
        for d, v in originais:
            d["demanda"] = 7 * (v or 0) + 1013

        def desfazer():
            for d, v in originais:
                d["demanda"] = v
        return desfazer

    def demanda(self, idx):
        return np.array([self.linhas[t]["demanda"] for t in idx], dtype=float)

    @property
    def treino(self):
        return [l for l, p in zip(self.linhas, self.partes) if p == "treino"]

    @property
    def validacao(self):
        return [l for l, p in zip(self.linhas, self.partes) if p == "validacao"]


def carregar_serie(seed, curso, caminho=None):
    """Lê `series/<curso>/serie_<seed>.csv`."""
    caminho = caminho or caminho_serie(seed, curso)
    with open(caminho, newline="", encoding="utf-8") as f:
        registros = list(csv.DictReader(f))
    return Serie(seed, curso, registros)


def _como_vetor(x, t, esperado):
    if isinstance(x, (int, float)):
        x = [x]
    try:
        v = [float(a) for a in x]
    except (TypeError, ValueError):
        raise ValueError(f"dia {t}: features devolveu {type(x).__name__}, "
                         f"esperado lista de números") from None
    if esperado is not None and len(v) != esperado:
        raise ValueError(f"dia {t}: {len(v)} features, mas o primeiro dia "
                         f"teve {esperado}. O tamanho precisa ser fixo")
    if not all(math.isfinite(a) for a in v):
        raise ValueError(f"dia {t}: feature não finita (nan ou inf): {v}")
    return v


def montar(serie, features, *partes, alvo="demanda"):
    """Monta (X, y, idx) para os dias das partes pedidas.

    alvo = "demanda"     y é a demanda do dia (regressão)
    alvo = "doca_extra"  y é 1 se a demanda passou da capacidade (classificação)
    """
    idx = serie.indices(*partes)
    linhas = []
    n = None
    for t in idx:
        v = _como_vetor(features(serie.hist(t), serie.dia(t)), t, n)
        n = len(v)
        linhas.append(v)
    X = np.array(linhas, dtype=float)
    y = serie.demanda(idx)
    if alvo == "doca_extra":
        y = (y > serie.capacidade).astype(float)
    elif alvo != "demanda":
        raise ValueError("alvo deve ser 'demanda' ou 'doca_extra'")
    return X, y, idx


def dividir(X, y, modo="temporal", frac_val=0.2, seed=0):
    """Separa uma parte de (X, y) para validação.

    modo = "temporal"   os últimos frac_val dos dias viram validação
    modo = "aleatorio"  dias sorteados viram validação
    """
    n = len(y)
    k = int(round(n * frac_val))
    if modo == "temporal":
        ordem = np.arange(n)
    elif modo == "aleatorio":
        ordem = np.random.default_rng(seed).permutation(n)
        ordem = np.concatenate([np.sort(ordem[:n - k]), np.sort(ordem[n - k:])])
    else:
        raise ValueError("modo deve ser 'temporal' ou 'aleatorio'")
    a, b = ordem[:n - k], ordem[n - k:]
    return X[a], y[a], X[b], y[b]
