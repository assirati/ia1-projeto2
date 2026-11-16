"""
Verifica sua submissão ANTES de enviar.

    python verificar_submissao.py <matricula> <curso>

Roda as mesmas checagens do placar, mas na sua série e em mais duas do
pool, treinando no treino e medindo na validação (o placar treina no
treino + validação e mede no teste oculto). Se algo falhar aqui, vai
falhar lá.

Este script NÃO diz se a sua previsão é boa. Diz se ela é válida.
"""

import argparse
import os
import sys

from engine.avaliacao import carregar, checar_codigo, validar_config, avaliar
from engine.config import LIMITES, PERFIL, atribuir, carregar_pools
from engine.dados import carregar_serie

OK = "  [ok]   "
ERRO = "  [ERRO] "
AVISO = "  [!]    "

ARQUIVO = os.path.join("submissao", "previsor.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matricula")
    ap.add_argument("curso", choices=sorted(PERFIL))
    args = ap.parse_args()

    if not os.path.exists(ARQUIVO):
        print(f"{ERRO}{ARQUIVO} não encontrado. Rode de dentro da pasta do repositório.")
        return 1

    print("CÓDIGO")
    problemas = checar_codigo(ARQUIVO)
    for p in problemas:
        print(ERRO + p)
    if problemas:
        print("\nSUBMISSÃO INVÁLIDA. Corrija os itens marcados com [ERRO].")
        return 1
    print(OK + "só módulos permitidos, sem acesso a arquivos")

    try:
        mod = carregar(ARQUIVO)
    except Exception as e:
        print(f"{ERRO}não consegui carregar {ARQUIVO}: {type(e).__name__}: {e}")
        return 1

    nome = getattr(mod, "NOME", "")
    if not isinstance(nome, str) or not nome.strip() or "COLOQUE" in nome:
        print(f"{ERRO}preencha a variável NOME")
        return 1
    print(OK + f"submissão de: {nome}")

    if not callable(getattr(mod, "features", None)):
        print(f"{ERRO}falta a função features(hist, dia)")
        return 1
    if hasattr(mod, "decidir") and not callable(mod.decidir):
        print(f"{ERRO}decidir existe mas não é uma função")
        return 1

    cfg, probs = validar_config(getattr(mod, "CONFIG", None))
    for p in probs:
        print(ERRO + p)
    if probs:
        print("\nSUBMISSÃO INVÁLIDA. Corrija os itens marcados com [ERRO].")
        return 1
    print(OK + "CONFIG válida: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    if hasattr(mod, "decidir"):
        print(OK + "decidir presente: será usada nas colunas de custo")

    pools = carregar_pools()
    minha = atribuir(args.matricula, args.curso, pools)
    outras = [s for s in pools[args.curso][:5] if s != minha][:2]

    print("\nSÉRIES (treino -> validação, 3 sementes por série)")
    falhou = False
    for seed in [minha] + outras:
        serie = carregar_serie(seed, args.curso)
        rot = f"série {seed}" + (" (a sua)" if seed == minha else "")

        # features precisa ser determinística: mesma entrada, mesma saída
        t = serie.indices("validacao")[0]
        try:
            a = list(mod.features(serie.hist(t), serie.dia(t)))
            b = list(mod.features(serie.hist(t), serie.dia(t)))
            if a != b:
                print(AVISO + f"{rot}: features devolveu valores diferentes para "
                              f"a mesma entrada. O placar não é reprodutível assim")
        except Exception:
            pass  # o erro aparece com detalhe em avaliar()

        r = avaliar(mod, cfg, serie, ("treino",), "validacao")
        if r["status"] == "falha":
            print(ERRO + f"{rot}: {r['detalhe']}")
            falhou = True
            continue
        if r["status"] == "censurada":
            print(AVISO + f"{rot}: passou de {LIMITES['limite_s']} s "
                          f"({r['ms'] / 1000:.0f} s). Será censurada no placar.")
            continue
        print(OK + f"{rot}: MSE {r['mse']:.0f}, RMSE relativo {r['rmse_rel']:.1f}%, "
                   f"viés {r['vies']:+.1f}%, "
                   f"custo {r['custo_100']:.1f} por 100 un., "
                   f"{r['n_features']} features, {r['ms'] / 1000:.1f} s")
        if r["ms"] > 500 * LIMITES["limite_s"]:
            print(AVISO + f"{rot}: usou mais da metade do limite de "
                          f"{LIMITES['limite_s']} s. A máquina do placar pode ser mais lenta")
        ws = r["rmse_rel_por_semente"]
        if max(ws) - min(ws) > 5:
            print(AVISO + f"{rot}: o RMSE relativo varia de {min(ws):.1f}% a {max(ws):.1f}% "
                          f"entre as sementes. Vocês sabem explicar por quê?")

    print()
    if falhou:
        print("SUBMISSÃO INVÁLIDA. Corrija os itens marcados com [ERRO].")
        return 1
    print("Submissão válida. Os avisos [!] não impedem o envio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
