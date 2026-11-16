"""
Rede neural (perceptron multicamada) com backpropagation escrito à mão.

    rede = MLP(camadas=[16], ativacao="relu", perda="mse")
    rede.treinar(X, y)                 # X: (n, m), y: (n,)
    p = rede.prever(X_novo)
    rede.historico["perda_treino"]     # uma entrada por época

Parâmetros
----------
camadas        neurônios de cada camada oculta. [] = sem camada oculta: a
               rede vira um modelo linear (com perda MSE, é a regressão linear)
ativacao       das camadas ocultas: identidade | sigmoide | tanh | relu | leaky_relu
perda          regressão:     mse | mae | huber | pinball
               classificação: entropia_cruzada | mse
quantil        para perda="pinball": a rede aprende o quantil q da demanda
delta          para perda="huber": onde a perda passa de quadrática a linear
               (nas unidades de y normalizado, se normalizar_y=True)
otimizador     sgd | momento | adam
taxa           taxa de aprendizado
epocas, lote   passadas pelo treino e tamanho do minilote
l2             penalização dos pesos (não dos vieses)
inicializacao  xavier | he | normal | zeros
escala_init    desvio padrão dos pesos quando inicializacao="normal"
normalizar_x   padroniza as entradas com as estatísticas do treino
normalizar_y   padroniza a demanda; a previsão volta para unidades
tarefa         regressao (saída identidade) | classificacao (saída sigmoide)
seed           semente da inicialização e da ordem dos minilotes

Instrumentação (rede.historico), uma entrada por época:
    perda_treino, perda_val   perda média, sem o termo l2. perda_val só
                              existe se treinar() receber X_val e y_val
    divergiu                  True se a perda virou nan ou infinito

Diagnósticos extras, não obrigatórios no projeto:
    norma_grad[c]             norma do gradiente dos pesos da camada c,
                              calculado no treino inteiro ao fim da época.
                              c = 0 é a primeira camada (junto da entrada)
    mortas[c]                 fração de neurônios da camada oculta c com
                              saída zero para TODOS os exemplos (ReLU morta)
    saturacao[c]              fração dos pares (exemplo, neurônio) da
                              camada oculta c em que a derivada da ativação
                              está abaixo de 5% do seu máximo

checar_gradiente(rede, X, y) compara o gradiente do backpropagation com
uma diferença finita, camada a camada.
"""

import numpy as np

from .modelos import Padronizador


# --------------------------------------------------------------- ativações
# Cada ativação: (f(z), f'(z) escrita em termos de z e de a = f(z), máximo de f')

def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


ATIVACOES = {
    "identidade": (lambda z: z, lambda z, a: np.ones_like(z), 1.0),
    "sigmoide": (_sig, lambda z, a: a * (1 - a), 0.25),
    "tanh": (np.tanh, lambda z, a: 1 - a * a, 1.0),
    "relu": (lambda z: np.maximum(z, 0.0),
             lambda z, a: (z > 0).astype(float), 1.0),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.01 * z),
                   lambda z, a: np.where(z > 0, 1.0, 0.01), 1.0),
}

PERDAS_REGRESSAO = ("mse", "mae", "huber", "pinball")
PERDAS_CLASSIFICACAO = ("entropia_cruzada", "mse")


# ------------------------------------------------------------------ perdas
# Cada perda devolve (valor médio, dL/d(saída) para cada exemplo).
# A divisão por n faz o gradiente ser o da perda MÉDIA do minilote.

def _perda(nome, s, y, quantil, delta):
    n = len(y)
    r = s - y  # resíduo: positivo = previu acima
    if nome == "mse":
        return float(np.mean(r ** 2)), 2 * r / n
    if nome == "mae":
        return float(np.mean(np.abs(r))), np.sign(r) / n
    if nome == "huber":
        a = np.abs(r)
        quad = a <= delta
        valor = np.where(quad, 0.5 * r ** 2, delta * (a - 0.5 * delta))
        grad = np.where(quad, r, delta * np.sign(r))
        return float(np.mean(valor)), grad / n
    if nome == "pinball":
        # abaixo do quantil (s < y) custa q; acima custa (1 - q)
        valor = np.where(r < 0, -quantil * r, (1 - quantil) * r)
        grad = np.where(r < 0, -quantil, 1 - quantil)
        return float(np.mean(valor)), grad / n
    raise ValueError(f"perda desconhecida: {nome}")


# ------------------------------------------------------------------- a rede

class MLP:
    def __init__(self, camadas=(16,), ativacao="relu", perda="mse",
                 quantil=0.5, delta=1.0, otimizador="adam", taxa=0.01,
                 epocas=200, lote=32, l2=0.0, inicializacao="he",
                 escala_init=1.0, normalizar_x=True, normalizar_y=True,
                 tarefa="regressao", seed=0):
        if ativacao not in ATIVACOES:
            raise ValueError(f"ativacao deve ser uma de {sorted(ATIVACOES)}")
        if tarefa not in ("regressao", "classificacao"):
            raise ValueError("tarefa deve ser 'regressao' ou 'classificacao'")
        validas = PERDAS_REGRESSAO if tarefa == "regressao" else PERDAS_CLASSIFICACAO
        if perda not in validas:
            raise ValueError(f"para {tarefa}, perda deve ser uma de {validas}")
        if otimizador not in ("sgd", "momento", "adam"):
            raise ValueError("otimizador deve ser sgd, momento ou adam")
        if inicializacao not in ("xavier", "he", "normal", "zeros"):
            raise ValueError("inicializacao deve ser xavier, he, normal ou zeros")
        if perda == "pinball" and not 0 < quantil < 1:
            raise ValueError("quantil deve estar entre 0 e 1")
        self.camadas = list(camadas)
        self.ativacao = ativacao
        self.perda = perda
        self.quantil = quantil
        self.delta = delta
        self.otimizador = otimizador
        self.taxa = taxa
        self.epocas = epocas
        self.lote = lote
        self.l2 = l2
        self.inicializacao = inicializacao
        self.escala_init = escala_init
        self.normalizar_x = normalizar_x
        self.normalizar_y = normalizar_y and tarefa == "regressao"
        self.tarefa = tarefa
        self.seed = seed
        self.W = None

    # ---------------------------------------------------------- preparação

    def preparar(self, X, y):
        """Ajusta as normalizações e sorteia os pesos iniciais."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.pad_x = Padronizador().ajustar(X) if self.normalizar_x else None
        if self.normalizar_y:
            self.y_media = float(y.mean())
            self.y_desvio = float(y.std()) or 1.0
        else:
            self.y_media, self.y_desvio = 0.0, 1.0
        self.rng = np.random.default_rng(self.seed)
        tamanhos = [X.shape[1]] + self.camadas + [1]
        self.W, self.b = [], []
        for n_in, n_out in zip(tamanhos[:-1], tamanhos[1:]):
            if self.inicializacao == "xavier":
                dp = np.sqrt(2.0 / (n_in + n_out))
            elif self.inicializacao == "he":
                dp = np.sqrt(2.0 / n_in)
            elif self.inicializacao == "normal":
                dp = self.escala_init
            else:
                dp = 0.0
            self.W.append(self.rng.normal(0, 1, (n_in, n_out)) * dp)
            self.b.append(np.zeros(n_out))
        self._estado = [dict() for _ in range(2 * len(self.W))]
        self._passo = 0
        return self

    def _x(self, X):
        X = np.asarray(X, dtype=float)
        return self.pad_x.aplicar(X) if self.pad_x is not None else X

    def _y(self, y):
        return (np.asarray(y, dtype=float) - self.y_media) / self.y_desvio

    # --------------------------------------------------- passo para frente

    def _frente(self, X):
        """Guarda as pré-ativações Z e as ativações A de cada camada.

        A[0] = X;  Z[c+1] = A[c]·W[c] + b[c];  A[c+1] = f(Z[c+1])
        A última camada usa identidade (regressão) ou sigmoide (classificação).
        """
        f = ATIVACOES[self.ativacao][0]
        A, Z = [X], [None]
        ultima = len(self.W) - 1
        for c, (W, b) in enumerate(zip(self.W, self.b)):
            z = A[-1] @ W + b
            Z.append(z)
            if c < ultima:
                A.append(f(z))
            elif self.tarefa == "classificacao":
                A.append(_sig(z))
            else:
                A.append(z)
        return Z, A

    # ------------------------------------------------------ backpropagation

    def _gradientes(self, X, y):
        """Perda média e gradientes dL/dW, dL/db de todas as camadas.

        A regra da cadeia, de trás para frente:
            delta = dL/dZ da camada atual
            dL/dW[c] = A[c]ᵀ · delta        (+ l2·W[c])
            dL/db[c] = soma de delta nos exemplos
            delta da camada anterior = (delta · W[c]ᵀ) ⊙ f'(Z[c])
        """
        Z, A = self._frente(X)
        s = A[-1][:, 0]
        if self.tarefa == "classificacao":
            p = np.clip(s, 1e-12, 1 - 1e-12)
            if self.perda == "entropia_cruzada":
                valor = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
                delta = ((s - y) / len(y))[:, None]  # sigmoide + entropia cruzada
            else:  # mse sobre a probabilidade
                valor = float(np.mean((s - y) ** 2))
                delta = (2 * (s - y) / len(y) * s * (1 - s))[:, None]
        else:
            valor, g = _perda(self.perda, s, y, self.quantil, self.delta)
            delta = g[:, None]  # saída identidade: dL/dZ = dL/ds

        df = ATIVACOES[self.ativacao][1]
        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        for c in range(len(self.W) - 1, -1, -1):
            gW[c] = A[c].T @ delta + self.l2 * self.W[c]
            gb[c] = delta.sum(axis=0)
            if c > 0:
                delta = (delta @ self.W[c].T) * df(Z[c], A[c])
        return valor, gW, gb, Z, A

    # ------------------------------------------------------------- otimização

    def _atualizar(self, gW, gb):
        self._passo += 1
        params = self.W + self.b
        grads = gW + gb
        for i, (p, g) in enumerate(zip(params, grads)):
            st = self._estado[i]
            if self.otimizador == "sgd":
                p -= self.taxa * g
            elif self.otimizador == "momento":
                v = st.get("v", np.zeros_like(p))
                v = 0.9 * v - self.taxa * g
                st["v"] = v
                p += v
            else:  # adam
                m = st.get("m", np.zeros_like(p))
                v = st.get("v", np.zeros_like(p))
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                st["m"], st["v"] = m, v
                mh = m / (1 - 0.9 ** self._passo)
                vh = v / (1 - 0.999 ** self._passo)
                p -= self.taxa * mh / (np.sqrt(vh) + 1e-8)

    def _diagnostico(self, Z, A):
        df = ATIVACOES[self.ativacao][1]
        dmax = ATIVACOES[self.ativacao][2]
        mortas, sat = [], []
        for c in range(1, len(self.W)):  # só camadas ocultas
            a = A[c]
            mortas.append(float(np.mean(np.all(a == 0, axis=0))))
            sat.append(float(np.mean(df(Z[c], a) < 0.05 * dmax)))
        return mortas, sat

    def perda_em(self, X, y):
        """Perda média (sem l2) num conjunto, nas unidades de treino."""
        Xn = self._x(X)
        yn = self._y(y) if self.tarefa == "regressao" else np.asarray(y, float)
        l2, self.l2 = self.l2, 0.0
        try:
            return self._gradientes(Xn, yn)[0]
        finally:
            self.l2 = l2

    # ------------------------------------------------------------------ treino

    def treinar(self, X, y, X_val=None, y_val=None):
        # divergência é um resultado, registrado em historico["divergiu"]:
        # os avisos de overflow do numpy só poluiriam a saída
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._treinar(X, y, X_val, y_val)

    def _treinar(self, X, y, X_val, y_val):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.preparar(X, y)
        Xn = self._x(X)
        yn = self._y(y) if self.tarefa == "regressao" else y
        tem_val = X_val is not None
        if tem_val:
            Xv = self._x(X_val)
            yv = self._y(y_val) if self.tarefa == "regressao" else np.asarray(y_val, float)

        H = self.historico = dict(perda_treino=[], perda_val=[], norma_grad=[],
                                  mortas=[], saturacao=[], divergiu=False)
        n = len(yn)
        for ep in range(self.epocas):
            ordem = self.rng.permutation(n)
            for i in range(0, n, self.lote):
                j = ordem[i:i + self.lote]
                _, gW, gb, _, _ = self._gradientes(Xn[j], yn[j])
                self._atualizar(gW, gb)

            # diagnóstico no treino inteiro, sem l2 na perda
            l2, self.l2 = self.l2, 0.0
            valor, gW, _, Z, A = self._gradientes(Xn, yn)
            self.l2 = l2
            H["perda_treino"].append(valor)
            H["norma_grad"].append([float(np.linalg.norm(g)) for g in gW])
            m, s = self._diagnostico(Z, A)
            H["mortas"].append(m)
            H["saturacao"].append(s)
            if not np.isfinite(valor) or not all(np.isfinite(w).all() for w in self.W):
                H["divergiu"] = True
                break
            if tem_val:
                l2, self.l2 = self.l2, 0.0
                vval = self._gradientes(Xv, yv)[0]
                self.l2 = l2
                H["perda_val"].append(vval)
        return self

    # -------------------------------------------------------------- previsão

    def prever(self, X):
        """Regressão: demanda prevista (em unidades). Classificação: P(y=1)."""
        with np.errstate(over="ignore", invalid="ignore"):
            _, A = self._frente(self._x(X))
        s = A[-1][:, 0]
        if self.tarefa == "regressao":
            return s * self.y_desvio + self.y_media
        return s

    def prever_classe(self, X, limiar=0.5):
        return (self.prever(X) >= limiar).astype(float)


# ---------------------------------------------------- checagem do gradiente

def checar_gradiente(rede, X, y, eps=1e-6, por_camada=20, seed=0):
    """Compara o gradiente do backpropagation com a diferença finita central

        (L(w + eps) - L(w - eps)) / (2·eps)

    em até `por_camada` pesos sorteados de cada camada. Devolve, por camada,
    o maior erro relativo |analítico - numérico| / (|analítico| + |numérico|).
    Valores da ordem de 1e-5 ou menores indicam gradiente correto; um erro
    no backpropagation costuma dar 1e-2 ou mais. Perdas e ativações com
    "bico" (mae, pinball, relu) podem dar erro maior quando algum exemplo
    cai exatamente no bico.

    Se a rede ainda não foi treinada, usa os pesos iniciais.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if rede.W is None:
        rede.preparar(X, y)
    Xn = rede._x(X)
    yn = rede._y(y) if rede.tarefa == "regressao" else y
    _, gW, _, _, _ = rede._gradientes(Xn, yn)
    rng = np.random.default_rng(seed)
    erros = []
    for c, W in enumerate(rede.W):
        pior = 0.0
        flat = W.reshape(-1)
        k = min(por_camada, flat.size)
        for i in rng.choice(flat.size, k, replace=False):
            orig = flat[i]
            flat[i] = orig + eps
            lp = rede._gradientes(Xn, yn)[0] + 0.5 * rede.l2 * sum(
                float((w ** 2).sum()) for w in rede.W)
            flat[i] = orig - eps
            lm = rede._gradientes(Xn, yn)[0] + 0.5 * rede.l2 * sum(
                float((w ** 2).sum()) for w in rede.W)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gW[c].reshape(-1)[i]
            den = abs(ana) + abs(num)
            if den > 0:
                pior = max(pior, abs(ana - num) / den)
        erros.append(pior)
    return erros
