"""
Avaliação de uma submissão ao placar. O mesmo código roda em dois lugares:

    verificar_submissao.py   treina no treino, mede na validação
    placar (professor)       treina no treino + validação, mede no teste oculto

Contrato do arquivo `submissao/previsor.py`:

    NOME = "nome do grupo"
    CONFIG = {...}                  # configuração da MLP (ver CONFIG_PADRAO)
    def features(hist, dia): ...    # lista de números, tamanho fixo
    def decidir(previsao, hist, dia): ...   # opcional: quanto estocar

O arquivo só pode importar módulos da lista MODULOS_PERMITIDOS e não pode
abrir arquivos, nem acessar atributos que começam com "_" (e outros que dão
acesso a frames). Além disso, avaliar() confere em tempo de execução que
as features de um dia não mudam quando a demanda daquele dia em diante
muda.
"""

import ast
import importlib.util
import math
import time
import traceback
import uuid

import numpy as np

from .config import CUSTO_FALTA, CUSTO_SOBRA, LIMITES
from .dados import _como_vetor, montar
from .metricas import custo_estoque, regressao
from .rede import ATIVACOES, MLP, PERDAS_REGRESSAO

MODULOS_PERMITIDOS = {"math", "statistics", "datetime", "itertools",
                      "functools", "collections", "bisect", "heapq"}

NOMES_PROIBIDOS = {"open", "eval", "exec", "compile", "__import__", "globals",
                   "locals", "vars", "getattr", "setattr", "delattr", "dir",
                   "breakpoint", "input", "help", "exit", "quit", "memoryview",
                   "type", "object", "super"}

# Atributos que dão acesso a frames (e, por eles, às variáveis de quem
# chamou) ou que acessam atributos por dentro de uma string de formatação.
ATRIBUTOS_PROIBIDOS = {"format", "format_map", "mro",
                       # statistics.sys existe, e sys.modules dá acesso a tudo
                       "sys", "os", "modules", "builtins", "random", "numbers",
                       "abc", "io", "subprocess", "importlib", "system", "popen"}
PREFIXOS_PROIBIDOS = ("_", "f_", "gi_", "cr_", "ag_", "tb_", "co_")

CONFIG_PADRAO = dict(
    camadas=[16], ativacao="relu", perda="mse", quantil=0.5, delta=1.0,
    otimizador="adam", taxa=0.01, epocas=200, lote=32, l2=0.0,
    inicializacao="he", normalizar=True, paciencia=None,
)

# Sementes de inicialização da rede. Cada série é treinada uma vez por
# semente, e a métrica reportada é a média.
SEMENTES = (0, 1, 2)


# ------------------------------------------------------------ código-fonte

def checar_codigo(caminho):
    """Lista de problemas do código-fonte, sem executá-lo."""
    try:
        with open(caminho, encoding="utf-8") as f:
            arvore = ast.parse(f.read(), filename=caminho)
    except SyntaxError as e:
        return [f"erro de sintaxe na linha {e.lineno}: {e.msg}"]
    problemas = []
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            for a in no.names:
                raiz = a.name.split(".")[0]
                if raiz not in MODULOS_PERMITIDOS:
                    problemas.append(f"linha {no.lineno}: import {a.name} não é permitido")
        elif isinstance(no, ast.ImportFrom):
            raiz = (no.module or "").split(".")[0]
            if no.level or raiz not in MODULOS_PERMITIDOS:
                problemas.append(f"linha {no.lineno}: from {no.module} import ... "
                                 f"não é permitido")
            for a in no.names:
                if a.name in ATRIBUTOS_PROIBIDOS or a.name.startswith("_") or a.name == "*":
                    problemas.append(f"linha {no.lineno}: from {no.module} import "
                                     f"{a.name} não é permitido")
        elif isinstance(no, ast.Name) and (no.id in NOMES_PROIBIDOS or (
                no.id.startswith("__") and no.id != "__name__")):
            problemas.append(f"linha {no.lineno}: uso de {no.id} não é permitido")
        elif isinstance(no, ast.Attribute) and (
                no.attr in ATRIBUTOS_PROIBIDOS
                or no.attr.startswith(PREFIXOS_PROIBIDOS)):
            problemas.append(f"linha {no.lineno}: atributo .{no.attr} não é permitido")
    return problemas


def carregar(caminho):
    """Importa o arquivo como um módulo novo (sem cache entre chamadas)."""
    nome = f"submissao_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- CONFIG

def validar_config(cfg):
    """Devolve (config completa, lista de problemas)."""
    if not isinstance(cfg, dict):
        return None, ["CONFIG precisa ser um dicionário"]
    problemas = []
    desconhecidas = set(cfg) - set(CONFIG_PADRAO)
    if desconhecidas:
        problemas.append(f"chaves desconhecidas em CONFIG: {sorted(desconhecidas)}")
    c = dict(CONFIG_PADRAO)
    c.update({k: v for k, v in cfg.items() if k in CONFIG_PADRAO})
    L = LIMITES

    def inteiro(v):
        return isinstance(v, int) and not isinstance(v, bool)

    def real(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    cam = c["camadas"]
    if not isinstance(cam, (list, tuple)) or not all(inteiro(n) for n in cam):
        problemas.append("camadas deve ser uma lista de inteiros, ex.: [16] ou []")
    else:
        if len(cam) > L["max_camadas"]:
            problemas.append(f"no máximo {L['max_camadas']} camadas ocultas")
        if any(n < 1 or n > L["max_neuronios"] for n in cam):
            problemas.append(f"cada camada oculta tem de 1 a {L['max_neuronios']} neurônios")
        c["camadas"] = list(cam)
    if c["ativacao"] not in ATIVACOES:
        problemas.append(f"ativacao deve ser uma de {sorted(ATIVACOES)}")
    if c["perda"] not in PERDAS_REGRESSAO:
        problemas.append(f"perda deve ser uma de {list(PERDAS_REGRESSAO)}")
    if not real(c["quantil"]) or not 0 < c["quantil"] < 1:
        problemas.append("quantil deve estar entre 0 e 1 (exclusive)")
    if not real(c["delta"]) or c["delta"] <= 0:
        problemas.append("delta deve ser positivo")
    if c["otimizador"] not in ("sgd", "momento", "adam"):
        problemas.append("otimizador deve ser sgd, momento ou adam")
    if not real(c["taxa"]) or not 0 < c["taxa"] <= 1:
        problemas.append("taxa deve estar em (0, 1]")
    if not inteiro(c["epocas"]) or not 1 <= c["epocas"] <= L["max_epocas"]:
        problemas.append(f"epocas deve ser inteiro de 1 a {L['max_epocas']}")
    if not inteiro(c["lote"]) or c["lote"] < L["min_lote"]:
        problemas.append(f"lote deve ser inteiro, no mínimo {L['min_lote']}")
    if not real(c["l2"]) or c["l2"] < 0:
        problemas.append("l2 deve ser um número >= 0")
    if c["inicializacao"] not in ("xavier", "he", "normal", "zeros"):
        problemas.append("inicializacao deve ser xavier, he, normal ou zeros")
    if not isinstance(c["normalizar"], bool):
        problemas.append("normalizar deve ser True ou False")
    if c["paciencia"] is not None and (not inteiro(c["paciencia"]) or c["paciencia"] < 1):
        problemas.append("paciencia deve ser None ou um inteiro >= 1")
    return c, problemas


def rede_da_config(c, seed):
    return MLP(camadas=c["camadas"], ativacao=c["ativacao"], perda=c["perda"],
               quantil=c["quantil"], delta=c["delta"], otimizador=c["otimizador"],
               taxa=c["taxa"], epocas=c["epocas"], lote=c["lote"], l2=c["l2"],
               inicializacao=c["inicializacao"], normalizar_x=c["normalizar"],
               normalizar_y=c["normalizar"], paciencia=c["paciencia"], seed=seed)


# --------------------------------------------------------------- avaliação

def _resumo_erro():
    return traceback.format_exc(limit=2).strip().splitlines()[-1]


def avaliar(mod, config, serie, partes_treino, parte_alvo, sementes=SEMENTES,
            limite_s=LIMITES["limite_s"], falta=CUSTO_FALTA, sobra=CUSTO_SOBRA):
    """Treina a rede da submissão e mede na parte alvo. Nunca lança exceção:
    o resultado tem status 'ok', 'falha' ou 'censurada'."""
    reg = {"seed": serie.seed}
    t0 = time.perf_counter()
    try:
        X, y, _ = montar(serie, mod.features, *partes_treino)
        Xa, ya, ia = montar(serie, mod.features, parte_alvo)
    except Exception:
        reg.update(status="falha", detalhe=f"features: {_resumo_erro()}")
        return reg
    if X.shape[1] != Xa.shape[1]:
        reg.update(status="falha", detalhe="features mudou de tamanho entre "
                                           "o treino e a previsão")
        return reg
    # As features do dia t não podem depender da demanda de t em diante.
    # Troca o futuro de alguns dias sorteados e confere que nada muda. Pega
    # qualquer leitura do futuro, por qualquer caminho.
    rng = np.random.default_rng(serie.seed)
    for k in sorted(rng.choice(len(ia), size=min(5, len(ia)), replace=False)):
        t = ia[k]
        desfazer = serie.trocar_futuro(t)
        try:
            v = np.array(_como_vetor(mod.features(serie.hist(t), serie.dia(t)), t, None))
            igual = v.shape == Xa[k].shape and np.allclose(v, Xa[k], rtol=1e-9, atol=1e-9)
        except Exception:
            igual = False
        finally:
            desfazer()
        if not igual:
            reg.update(status="falha", detalhe=f"features do dia {serie.linhas[t]['data']} "
                                               f"mudam quando a demanda daquele dia em "
                                               f"diante muda: a função lê o futuro")
            return reg
    if X.shape[1] > LIMITES["max_features"]:
        reg.update(status="falha", detalhe=f"{X.shape[1]} features, o máximo "
                                           f"é {LIMITES['max_features']}")
        return reg
    reg["n_features"] = int(X.shape[1])

    decidir = getattr(mod, "decidir", None)
    por_semente = []
    for s in sementes:
        try:
            rede = rede_da_config(config, s).treinar(X, y)
        except Exception:
            reg.update(status="falha", detalhe=f"treino: {_resumo_erro()}")
            return reg
        p = rede.prever(Xa)
        if rede.historico["divergiu"] or not np.all(np.isfinite(p)):
            reg.update(status="falha", detalhe=f"a rede divergiu (semente {s}): "
                                               f"a perda virou nan ou infinito")
            return reg
        p = np.maximum(p, 0.0)  # previsão negativa vira zero
        if decidir is None:
            q = p
        else:
            try:
                q = np.array([float(decidir(float(pi), serie.hist(t), serie.dia(t)))
                              for pi, t in zip(p, ia)])
            except Exception:
                reg.update(status="falha", detalhe=f"decidir: {_resumo_erro()}")
                return reg
            if not np.all(np.isfinite(q)):
                reg.update(status="falha", detalhe="decidir devolveu nan ou infinito")
                return reg
        m = regressao(ya, p)
        m.update(custo_estoque(ya, q, falta, sobra))
        m["epocas_usadas"] = len(rede.historico["perda_treino"])
        por_semente.append(m)
        if time.perf_counter() - t0 > limite_s:
            reg.update(status="censurada",
                       ms=round(1000 * (time.perf_counter() - t0)))
            return reg

    chaves = ("wape", "mae", "rmse", "vies", "custo_100", "atendimento",
              "dias_com_falta", "epocas_usadas")
    for k in chaves:
        reg[k] = round(float(np.mean([m[k] for m in por_semente])), 3)
    reg["wape_por_semente"] = [round(m["wape"], 3) for m in por_semente]
    reg["status"] = "ok"
    reg["ms"] = round(1000 * (time.perf_counter() - t0))
    return reg
