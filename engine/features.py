"""
Funções de features de referência. Cada uma recebe (hist, dia) e devolve
uma lista de números. Ver engine/dados.py para o que há em cada dia.

Copiem e modifiquem à vontade. O arquivo submetido ao placar não pode
importar este módulo: copiem o código que quiserem para dentro dele.
"""


def _media(xs):
    return sum(xs) / len(xs) if xs else 0.0


def calendario(hist, dia):
    """Só o calendário: dia da semana (one-hot), feriado, véspera, evento."""
    semana = [1.0 if dia["dia_semana"] == k else 0.0 for k in range(7)]
    return semana + [dia["feriado"], dia["vespera_feriado"], dia["evento"]]


def defasagens(hist, dia):
    """Só o passado da própria demanda: ontem, semana passada, médias."""
    d = [l["demanda"] for l in hist[-28:]]
    return [d[-1], d[-7], _media(d[-7:]), _media(d)]


def basicas(hist, dia):
    """Calendário + promoção + temperatura + defasagens. É o ponto de partida."""
    return (calendario(hist, dia)
            + [dia["promocao"], dia["desconto"], dia["temperatura"]]
            + defasagens(hist, dia))
