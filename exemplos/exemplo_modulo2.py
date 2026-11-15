"""
Módulo 2: perceptron e rede neural na série do grupo. Grava
dados/modulo2.csv e dados/modulo2_historico.json (curvas por época).

    python exemplos/exemplo_modulo2.py <matricula> <curso>

Isto NÃO é a entrega. É o esqueleto para vocês adaptarem.

Cada configuração da rede é treinada com 3 sementes. Rede neural não é
determinística entre sementes: uma linha por semente, sem filtrar.
Leva cerca de 1 minuto.
"""

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from engine import features as F
from engine.config import atribuir
from engine.dados import carregar_serie, montar
from engine.metricas import classificacao, custo_estoque, regressao
from engine.modelos import Perceptron, RegressaoLogistica
from engine.rede import MLP, checar_gradiente

SEMENTES = (0, 1, 2)
CAMPOS = ["experimento", "camadas", "ativacao", "perda", "quantil", "taxa",
          "semente", "wape", "vies", "custo_100", "atendimento", "acuracia",
          "revocacao", "epocas_usadas", "perda_treino_final", "perda_val_final",
          "grad_primeira", "grad_ultima", "mortas", "divergiu", "ms"]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    matricula, curso = sys.argv[1], sys.argv[2]
    seed = atribuir(matricula, curso)
    s = carregar_serie(seed, curso)
    print(f"série {seed} ({curso})\n")

    Xtr, ytr, _ = montar(s, F.basicas, "treino")
    Xva, yva, _ = montar(s, F.basicas, "validacao")
    linhas, curvas = [], {}

    def rodar(experimento, **kw):
        for sem in SEMENTES:
            t0 = time.perf_counter()
            rede = MLP(seed=sem, **kw).treinar(Xtr, ytr, Xva, yva)
            ms = 1000 * (time.perf_counter() - t0)
            H = rede.historico
            p = rede.prever(Xva)
            ok = not H["divergiu"]
            m = regressao(yva, p) if ok else {}
            c = custo_estoque(yva, p) if ok else {}
            linhas.append(dict(
                experimento=experimento, camadas=str(kw.get("camadas", [16])),
                ativacao=kw.get("ativacao", "relu"), perda=kw.get("perda", "mse"),
                quantil=kw.get("quantil", ""), taxa=kw.get("taxa", 0.01),
                semente=sem, wape=round(m.get("wape", float("nan")), 3),
                vies=round(m.get("vies", float("nan")), 3),
                custo_100=round(c.get("custo_100", float("nan")), 3),
                atendimento=round(c.get("atendimento", float("nan")), 3),
                epocas_usadas=len(H["perda_treino"]),
                perda_treino_final=round(H["perda_treino"][-1], 5),
                perda_val_final=round(H["perda_val"][-1], 5) if H["perda_val"] else "",
                grad_primeira=round(H["norma_grad"][0][0], 6),
                grad_ultima=round(H["norma_grad"][0][-1], 6),
                mortas=round(sum(H["mortas"][-1]) / max(1, len(H["mortas"][-1])), 3),
                divergiu=H["divergiu"], ms=round(ms)))
            curvas[f"{experimento}|{kw}|{sem}"] = {
                k: H[k] for k in ("perda_treino", "perda_val", "norma_grad", "mortas")}
            if sem == SEMENTES[0]:
                desc = ", ".join(f"{k}={v}" for k, v in kw.items())
                txt = (f"WAPE {m['wape']:5.1f}%  custo {c['custo_100']:6.1f}"
                       if ok else "DIVERGIU")
                print(f"  {experimento:<14}{desc[:62]:<64}{txt}")

    print("Ativação x perda (camadas=[16], 200 épocas)")
    for at in ("identidade", "sigmoide", "tanh", "relu"):
        for perda in ("mse", "mae", "huber"):
            rodar("ativ_perda", camadas=[16], ativacao=at, perda=perda)

    print("\nPerda pinball (o quantil importa para o estoque)")
    for q in (0.5, 0.8):
        rodar("pinball", camadas=[16], perda="pinball", quantil=q)

    print("\nTaxa de aprendizado (sgd)")
    for taxa in (0.001, 0.01, 0.1, 1.0):
        rodar("taxa", camadas=[16], otimizador="sgd", taxa=taxa)

    print("\nProfundidade: 4 camadas de 16, inicialização xavier")
    for at in ("sigmoide", "relu"):
        rodar("profundidade", camadas=[16] * 4, ativacao=at, inicializacao="xavier")

    print("\nChecagem do gradiente (maior erro relativo por camada)")
    for at, perda in (("tanh", "mse"), ("relu", "mae")):
        rede = MLP(camadas=[8, 8], ativacao=at, perda=perda)
        err = checar_gradiente(rede, Xtr[:64], ytr[:64])
        print(f"  {at:<6}{perda:<6}" + "  ".join(f"{e:.1e}" for e in err))

    print(f"\nDoca extra: perceptron x regressão logística (validação)")
    Ctr, ctr, _ = montar(s, F.basicas, "treino", alvo="doca_extra")
    Cva, cva, _ = montar(s, F.basicas, "validacao", alvo="doca_extra")
    for nome, mod in [("perceptron", Perceptron(epocas=50)),
                      ("perceptron_medio", Perceptron(epocas=50, media=True)),
                      ("logistica", RegressaoLogistica())]:
        t0 = time.perf_counter()
        mod.treinar(Ctr, ctr)
        m = classificacao(cva, mod.prever_classe(Cva))
        extra = ""
        if "erros" in mod.historico:
            e = mod.historico["erros"]
            extra = f"  erros na última época: {e[-1]} de {len(ctr)}"
        print(f"  {nome:<18}acurácia {m['acuracia']:5.1f}%  "
              f"revocação {m['revocacao']:5.1f}%{extra}")
        linhas.append(dict(experimento="doca_extra", camadas="", ativacao=nome,
                           perda="", acuracia=round(m["acuracia"], 3),
                           revocacao=round(m["revocacao"], 3),
                           ms=round(1000 * (time.perf_counter() - t0))))

    os.makedirs("dados", exist_ok=True)
    with open(os.path.join("dados", "modulo2.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(linhas)
    with open(os.path.join("dados", "modulo2_historico.json"), "w",
              encoding="utf-8") as f:
        json.dump(curvas, f)
    print("\ngravado: dados/modulo2.csv e dados/modulo2_historico.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
