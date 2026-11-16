"""
Métricas de avaliação.

Regressão (previsão de demanda)
    mse        erro quadrático médio, em unidades²
    rmse       raiz do MSE, de volta em unidades de demanda
    rmse_rel   RMSE dividido pela demanda média do período, em %. Dá para
               comparar séries de volumes diferentes. Ordena o placar de
               Computação.
    vies       soma(previsão - demanda) / soma da demanda, em %. Positivo =
               a previsão fica acima da demanda.

Estoque (Produção)
    custo_estoque   estoca q unidades para um dia de demanda d:
                    sobra  = max(q - d, 0)  custa CUSTO_SOBRA por unidade
                    falta  = max(d - q, 0)  custa CUSTO_FALTA por unidade
                    `custo_100` (custo por 100 unidades demandadas) ordena
                    o placar de Produção.

Classificação (doca extra)
    classificacao   acurácia, precisão, revocação, F1 e a matriz de confusão
    custo_doca      custo de abrir a doca extra sem precisar (falso positivo)
                    mais o de não abrir quando precisava (falso negativo)
"""

import numpy as np

from .config import CUSTO_DOCA_ABRIR, CUSTO_DOCA_FALTA, CUSTO_FALTA, CUSTO_SOBRA


def _arr(a):
    return np.asarray(a, dtype=float)


def mse(y, p):
    return float(np.mean((_arr(p) - _arr(y)) ** 2))


def rmse(y, p):
    return float(np.sqrt(mse(y, p)))


def rmse_rel(y, p):
    return float(100 * rmse(y, p) / np.mean(_arr(y)))


def vies(y, p):
    y, p = _arr(y), _arr(p)
    return float(100 * np.sum(p - y) / np.sum(y))


def regressao(y, p):
    return dict(mse=mse(y, p), rmse=rmse(y, p), rmse_rel=rmse_rel(y, p),
                vies=vies(y, p))


def custo_estoque(y, q, falta=CUSTO_FALTA, sobra=CUSTO_SOBRA):
    """Custo de estocar q[t] unidades para a demanda y[t], dia a dia."""
    y, q = _arr(y), np.maximum(_arr(q), 0.0)
    un_sobra = np.maximum(q - y, 0.0)
    un_falta = np.maximum(y - q, 0.0)
    custo = sobra * un_sobra.sum() + falta * un_falta.sum()
    return dict(
        custo=float(custo),
        custo_100=float(100 * custo / y.sum()),
        unidades_sobra=float(un_sobra.sum()),
        unidades_falta=float(un_falta.sum()),
        dias_com_falta=float(100 * np.mean(un_falta > 0)),  # % dos dias
        atendimento=float(100 * (1 - un_falta.sum() / y.sum())),  # % da demanda
    )


def quantil_critico(falta=CUSTO_FALTA, sobra=CUSTO_SOBRA):
    """Fração da demanda que o estoque ótimo deve cobrir (problema do jornaleiro)."""
    return falta / (falta + sobra)


def classificacao(y, c):
    """y e c com 0/1 (c = classe prevista)."""
    y, c = _arr(y).astype(int), _arr(c).astype(int)
    vp = int(np.sum((c == 1) & (y == 1)))
    fp = int(np.sum((c == 1) & (y == 0)))
    fn = int(np.sum((c == 0) & (y == 1)))
    vn = int(np.sum((c == 0) & (y == 0)))
    prec = vp / (vp + fp) if vp + fp else 0.0
    rev = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * prec * rev / (prec + rev) if prec + rev else 0.0
    return dict(acuracia=100 * (vp + vn) / len(y), precisao=100 * prec,
                revocacao=100 * rev, f1=100 * f1, vp=vp, fp=fp, fn=fn, vn=vn,
                positivos=100 * float(np.mean(y)))


def custo_doca(y, c, abrir=CUSTO_DOCA_ABRIR, falta=CUSTO_DOCA_FALTA):
    """Custo da decisão de abrir a doca extra (c = 1 abre), dia a dia.
    Abrir quando precisa não tem custo adicional aqui: é a operação normal."""
    y, c = _arr(y).astype(int), _arr(c).astype(int)
    fp = int(np.sum((c == 1) & (y == 0)))
    fn = int(np.sum((c == 0) & (y == 1)))
    total = abrir * fp + falta * fn
    return dict(custo=float(total), por_mes=float(total * 30 / len(y)), fp=fp, fn=fn)
