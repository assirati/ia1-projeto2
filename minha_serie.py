"""
Mostra a série de demanda do grupo.

    python minha_serie.py <matricula> <producao|computacao>

Use a matrícula do integrante mais velho do grupo.
"""

import argparse
import sys
from collections import defaultdict

from engine.config import CUSTO_FALTA, CUSTO_SOBRA, atribuir, caminho_serie
from engine.dados import carregar_serie


def br(x, casas=1):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matricula")
    ap.add_argument("curso", choices=["producao", "computacao"])
    args = ap.parse_args()

    seed = atribuir(args.matricula, args.curso)
    s = carregar_serie(seed, args.curso)
    print(f"matrícula {args.matricula}, curso {args.curso}: série {seed}")
    print(f"arquivo: {caminho_serie(seed, args.curso)}\n")

    print(f"{'parte':<16}{'período':<26}{'dias':>5}{'média':>9}{'mín':>6}{'máx':>6}")
    for parte in ("treino", "validacao"):
        ls = [l for l, p in zip(s.linhas, s.partes) if p == parte]
        d = [l["demanda"] for l in ls]
        per = f"{ls[0]['data']} a {ls[-1]['data']}"
        print(f"{parte:<16}{per:<26}{len(ls):>5}{br(sum(d) / len(d)):>9}"
              f"{min(d):>6}{max(d):>6}")
    print(f"{'teste (oculto)':<16}{'2025-10-01 a 2025-12-31':<26}{92:>5}"
          f"{'?':>9}{'?':>6}{'?':>6}")

    trein = [l for l, p in zip(s.linhas, s.partes) if p == "treino"]
    val = [l for l, p in zip(s.linhas, s.partes) if p == "validacao"]
    def frac(ls):
        return 100 * sum(l["demanda"] > s.capacidade for l in ls) / len(ls)
    print(f"\ncapacidade da doca normal: {s.capacidade} unidades/dia")
    print(f"  dias acima dela (doca extra): {br(frac(trein))}% do treino, "
          f"{br(frac(val))}% da validação")
    print(f"dias de promoção: {sum(l['promocao'] for l in trein)} no treino, "
          f"{sum(l['promocao'] for l in val)} na validação")
    print(f"custo do estoque: falta R$ {br(CUSTO_FALTA, 2)} por unidade, "
          f"sobra R$ {br(CUSTO_SOBRA, 2)} por unidade")

    print("\ndemanda média por dia da semana (treino):")
    sem = defaultdict(list)
    for l in trein:
        sem[l["dia_semana"]].append(l["demanda"])
    nomes = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
    print("  " + "  ".join(f"{nomes[k]} {br(sum(v) / len(v), 0)}"
                           for k, v in sorted(sem.items())))

    print("\ndemanda média por mês (cada # = 5 unidades/dia):")
    mes = defaultdict(list)
    for l in s.linhas:
        mes[l["data"][:7]].append(l["demanda"])
    for m, v in mes.items():
        media = sum(v) / len(v)
        print(f"  {m}  {'#' * int(round(media / 5)):<60} {br(media, 0)}")

    print("\nprimeiros dias do arquivo:")
    cols = ["data", "dia_semana", "feriado", "vespera_feriado", "evento",
            "promocao", "desconto", "preco", "temperatura", "demanda"]
    print("  " + " ".join(f"{c[:10]:>10}" for c in cols))
    for l in s.linhas[:8]:
        print("  " + " ".join(f"{str(l[c]):>10}" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
