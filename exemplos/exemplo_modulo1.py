"""
Módulo 1: modelos de referência, regressão linear, k-NN e regressão
logística na série do grupo. Grava dados/modulo1.csv.

    python exemplos/exemplo_modulo1.py <matricula> <curso>

Isto NÃO é a entrega. É o esqueleto para vocês adaptarem: troquem as
features, os parâmetros e as divisões, e perguntem por que os números
saíram assim.

Tudo é medido na VALIDAÇÃO (01/07 a 30/09/2025), treinando no treino.
As linhas com divisao="aleatorio" ou "temporal" usam só o treino: uma parte
dele vira validação interna, sorteada ou do fim do período.
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from engine import features as F
from engine.config import atribuir
from engine.dados import carregar_serie, dividir, montar
from engine.metricas import classificacao, custo_doca, regressao
from engine.modelos import (KNN, RegressaoLinear, RegressaoLogistica, ingenuo,
                            media_movel, prever_regra, sazonal)

CAMPOS = ["tarefa", "modelo", "parametro", "features", "divisao", "wape", "mae",
          "rmse", "vies", "acuracia", "precisao", "revocacao", "f1",
          "custo_doca_mes", "ms"]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    matricula, curso = sys.argv[1], sys.argv[2]
    seed = atribuir(matricula, curso)
    s = carregar_serie(seed, curso)
    print(f"série {seed} ({curso}), capacidade da doca normal = {s.capacidade}\n")
    linhas = []

    def reg(modelo, parametro, feats, divisao, y, p, ms):
        m = regressao(y, p)
        linhas.append(dict(tarefa="demanda", modelo=modelo, parametro=parametro,
                           features=feats, divisao=divisao, ms=round(ms, 1),
                           **{k: round(v, 3) for k, v in m.items()}))
        print(f"  {modelo:<18}{parametro:<10}{divisao:<12}"
              f"WAPE {m['wape']:5.1f}%  viés {m['vies']:+6.1f}%")

    print("Previsão de demanda (validação)")
    for nome, regra in [("ingenuo", ingenuo), ("sazonal", sazonal),
                        ("media_movel", media_movel(7)), ("media_movel", media_movel(28))]:
        t0 = time.perf_counter()
        p, y, _ = prever_regra(s, regra, "validacao")
        par = regra.__name__.split("_")[-1] if nome == "media_movel" else ""
        reg(nome, par, "-", "validacao", y, p, 1000 * (time.perf_counter() - t0))

    Xtr, ytr, _ = montar(s, F.basicas, "treino")
    Xva, yva, _ = montar(s, F.basicas, "validacao")

    for l2 in (0.0, 1.0, 10.0):
        t0 = time.perf_counter()
        p = RegressaoLinear(l2=l2).treinar(Xtr, ytr).prever(Xva)
        reg("linear", f"l2={l2:g}", "basicas", "validacao", yva, p,
            1000 * (time.perf_counter() - t0))

    for k in (1, 5, 25):
        t0 = time.perf_counter()
        p = KNN(k=k).treinar(Xtr, ytr).prever(Xva)
        reg("knn", f"k={k}", "basicas", "validacao", yva, p,
            1000 * (time.perf_counter() - t0))

    print("\nValidação interna, só dentro do treino (20% separados)")
    for modo in ("aleatorio", "temporal"):
        a, b, c, d = dividir(Xtr, ytr, modo=modo, frac_val=0.2, seed=0)
        for nome, mod in [("linear", RegressaoLinear()), ("knn", KNN(k=5))]:
            t0 = time.perf_counter()
            p = mod.treinar(a, b).prever(c)
            reg(nome, "k=5" if nome == "knn" else "l2=0", "basicas", modo, d, p,
                1000 * (time.perf_counter() - t0))

    print(f"\nDoca extra: a demanda passa de {s.capacidade}? (validação)")
    Ctr, ctr, _ = montar(s, F.basicas, "treino", alvo="doca_extra")
    Cva, cva, _ = montar(s, F.basicas, "validacao", alvo="doca_extra")

    def cls(modelo, parametro, y, c, ms):
        m = classificacao(y, c)
        cd = custo_doca(y, c)
        linhas.append(dict(tarefa="doca_extra", modelo=modelo, parametro=parametro,
                           features="basicas", divisao="validacao", ms=round(ms, 1),
                           custo_doca_mes=round(cd["por_mes"], 2),
                           **{k: round(m[k], 3) for k in
                              ("acuracia", "precisao", "revocacao", "f1")}))
        print(f"  {modelo:<18}{parametro:<14}acurácia {m['acuracia']:5.1f}%  "
              f"precisão {m['precisao']:5.1f}%  revocação {m['revocacao']:5.1f}%  "
              f"custo R$ {cd['por_mes']:7.0f}/mês")

    cls("sempre_nao", "", cva, 0 * cva, 0.0)
    for limiar in (0.5, 0.25, 0.1):
        t0 = time.perf_counter()
        lg = RegressaoLogistica().treinar(Ctr, ctr)
        cls("logistica", f"limiar={limiar}", cva, lg.prever_classe(Cva, limiar),
            1000 * (time.perf_counter() - t0))

    os.makedirs("dados", exist_ok=True)
    with open(os.path.join("dados", "modulo1.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(linhas)
    print("\ngravado: dados/modulo1.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
