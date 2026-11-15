"""
Modelos de referência (Unidade 2) e o perceptron (Unidade 3).

Regras de previsão, sem treino. Recebem (hist, dia) como as features:
    ingenuo        amanhã = hoje
    sazonal        amanhã = o mesmo dia da semana passada
    media_movel(k) amanhã = média dos últimos k dias
    prever_regra(serie, regra, *partes) aplica uma regra a uma parte da série

Modelos treinados. Todos têm .treinar(X, y) e .prever(X):
    RegressaoLinear(l2=0.0)                 mínimos quadrados; l2 > 0 é ridge
    KNN(k=5, pesos="uniforme")              média dos k vizinhos mais próximos
    RegressaoLogistica(taxa, epocas, l2)    classificação, descida de gradiente
    Perceptron(taxa, epocas, media=False)   classificação, regra de Rosenblatt

Todos padronizam as entradas (média 0, desvio 1) com as estatísticas do
treino, a menos que se passe normalizar=False.
"""

import numpy as np


# ------------------------------------------------------------------ regras

def ingenuo(hist, dia):
    return hist[-1]["demanda"]


def sazonal(hist, dia):
    return hist[-7]["demanda"]


def media_movel(k):
    def regra(hist, dia):
        d = [l["demanda"] for l in hist[-k:]]
        return sum(d) / len(d)
    regra.__name__ = f"media_movel_{k}"
    return regra


def prever_regra(serie, regra, *partes):
    """Devolve (previsões, demanda real, índices) para as partes pedidas."""
    idx = serie.indices(*partes)
    p = np.array([float(regra(serie.hist(t), serie.dia(t))) for t in idx])
    return p, serie.demanda(idx), idx


# -------------------------------------------------------------- utilitário

class Padronizador:
    """z = (x - média) / desvio, com as estatísticas do conjunto de ajuste."""

    def ajustar(self, X):
        X = np.asarray(X, dtype=float)
        self.media = X.mean(axis=0)
        self.desvio = X.std(axis=0)
        self.desvio[self.desvio == 0] = 1.0  # coluna constante
        return self

    def aplicar(self, X):
        return (np.asarray(X, dtype=float) - self.media) / self.desvio


class _Base:
    def __init__(self, normalizar=True):
        self.normalizar = normalizar
        self.pad = None

    def _entrada(self, X, ajustar=False):
        X = np.asarray(X, dtype=float)
        if not self.normalizar:
            return X
        if ajustar:
            self.pad = Padronizador().ajustar(X)
        return self.pad.aplicar(X)


# --------------------------------------------------------------- regressão

class RegressaoLinear(_Base):
    """y = X·w + b, minimizando o erro quadrático (+ l2·|w|² se l2 > 0)."""

    def __init__(self, l2=0.0, normalizar=True):
        super().__init__(normalizar)
        self.l2 = l2

    def treinar(self, X, y):
        Z = self._entrada(X, ajustar=True)
        y = np.asarray(y, dtype=float)
        A = np.hstack([Z, np.ones((len(Z), 1))])
        if self.l2 > 0:
            R = np.sqrt(self.l2) * np.eye(A.shape[1])
            R[-1, -1] = 0.0  # o intercepto não é penalizado
            A = np.vstack([A, R])
            y = np.concatenate([y, np.zeros(A.shape[1])])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.w, self.b = coef[:-1], coef[-1]
        return self

    def prever(self, X):
        return self._entrada(X) @ self.w + self.b


class KNN(_Base):
    """Média da demanda dos k dias de treino mais parecidos (distância euclidiana).

    pesos = "uniforme"   média simples
    pesos = "distancia"  média ponderada por 1/distância
    """

    def __init__(self, k=5, pesos="uniforme", normalizar=True):
        super().__init__(normalizar)
        if pesos not in ("uniforme", "distancia"):
            raise ValueError("pesos deve ser 'uniforme' ou 'distancia'")
        self.k, self.pesos = k, pesos

    def treinar(self, X, y):
        self.Xt = self._entrada(X, ajustar=True)
        self.yt = np.asarray(y, dtype=float)
        return self

    def prever(self, X):
        Z = self._entrada(X)
        d2 = ((Z[:, None, :] - self.Xt[None, :, :]) ** 2).sum(axis=2)
        k = min(self.k, len(self.yt))
        viz = np.argpartition(d2, k - 1, axis=1)[:, :k]
        yv = self.yt[viz]
        if self.pesos == "uniforme":
            return yv.mean(axis=1)
        w = 1.0 / (np.sqrt(np.take_along_axis(d2, viz, axis=1)) + 1e-9)
        return (w * yv).sum(axis=1) / w.sum(axis=1)


# ----------------------------------------------------------- classificação

def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class RegressaoLogistica(_Base):
    """P(y=1) = sigmoide(X·w + b), treinada por descida de gradiente na
    entropia cruzada. prever() devolve a probabilidade; prever_classe()
    compara com o limiar."""

    def __init__(self, taxa=0.1, epocas=500, l2=0.0, normalizar=True):
        super().__init__(normalizar)
        self.taxa, self.epocas, self.l2 = taxa, epocas, l2

    def treinar(self, X, y):
        Z = self._entrada(X, ajustar=True)
        y = np.asarray(y, dtype=float)
        n, m = Z.shape
        self.w, self.b = np.zeros(m), 0.0
        self.historico = {"perda": []}
        for _ in range(self.epocas):
            p = _sigmoide(Z @ self.w + self.b)
            g = p - y
            self.w -= self.taxa * (Z.T @ g / n + self.l2 * self.w)
            self.b -= self.taxa * g.mean()
            pc = np.clip(p, 1e-12, 1 - 1e-12)
            self.historico["perda"].append(
                float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
        return self

    def prever(self, X):
        return _sigmoide(self._entrada(X) @ self.w + self.b)

    def prever_classe(self, X, limiar=0.5):
        return (self.prever(X) >= limiar).astype(float)


class Perceptron(_Base):
    """Perceptron de Rosenblatt: ajusta os pesos só quando erra.

        se y·(w·x + b) <= 0:   w += taxa·y·x,  b += taxa·y     (y em {-1, +1})

    Converge se as classes forem linearmente separáveis. Se não forem, o
    número de erros por época não chega a zero. historico["erros"] guarda
    esse número época a época. Com media=True, prevê com a média dos pesos
    ao longo do treino (perceptron médio).
    """

    def __init__(self, taxa=1.0, epocas=50, media=False, seed=0, normalizar=True):
        super().__init__(normalizar)
        self.taxa, self.epocas, self.media, self.seed = taxa, epocas, media, seed

    def treinar(self, X, y):
        Z = self._entrada(X, ajustar=True)
        s = np.where(np.asarray(y) > 0.5, 1.0, -1.0)
        n, m = Z.shape
        w, b = np.zeros(m), 0.0
        sw, sb, cont = np.zeros(m), 0.0, 0
        rng = np.random.default_rng(self.seed)
        self.historico = {"erros": []}
        for _ in range(self.epocas):
            erros = 0
            for i in rng.permutation(n):
                if s[i] * (Z[i] @ w + b) <= 0:
                    w += self.taxa * s[i] * Z[i]
                    b += self.taxa * s[i]
                    erros += 1
                sw += w
                sb += b
                cont += 1
            self.historico["erros"].append(erros)
            if erros == 0:
                break
        if self.media:
            self.w, self.b = sw / cont, sb / cont
        else:
            self.w, self.b = w, b
        return self

    def prever(self, X):
        """Escore w·x + b. Positivo = classe 1."""
        return self._entrada(X) @ self.w + self.b

    def prever_classe(self, X, limiar=0.0):
        return (self.prever(X) > limiar).astype(float)
