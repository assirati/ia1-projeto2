"""
Configuração central. Importado tanto pelas ferramentas do aluno quanto
pelas do professor, para que não haja duas versões da mesma constante.
"""

import hashlib
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Custos do estoque (R$ por unidade). O que sobra no fim do dia é
# descartado (produto refrigerado); o que falta é venda perdida.
CUSTO_FALTA = 4.0
CUSTO_SOBRA = 1.0

# Custos da doca extra (R$ por dia). Abrir sem precisar: equipe parada.
# Precisar e não ter aberto: caminhões esperando, horas extras.
CUSTO_DOCA_ABRIR = 300.0
CUSTO_DOCA_FALTA = 1500.0

# O que cada curso usa para ordenar o placar. As duas colunas aparecem
# para os dois cursos.
PERFIL = {
    "producao": dict(metrica="custo_100"),
    "computacao": dict(metrica="rmse_rel"),
}

# Quantas séries do pool o placar usa
N_PLACAR = 30

# Os primeiros dias da série só servem de histórico: nunca são alvo de
# previsão, nem no treino. Lags de até 28 dias sempre existem.
AQUECIMENTO = 28

# Limites do contrato de submissão (ver ENUNCIADO, seção 5)
LIMITES = dict(
    max_camadas=4,      # camadas ocultas
    max_neuronios=64,   # por camada oculta
    max_epocas=500,
    min_lote=8,
    max_features=100,
    limite_s=60,        # por série, somando features e treino
)


def carregar_pools(path=None):
    """Lê a lista de séries aprovadas. O arquivo público não tem baseline."""
    if path is None:
        for nome in ("pools_publico.json", "pools.json"):
            p = os.path.join(RAIZ, nome)
            if os.path.exists(p):
                path = p
                break
    if path is None:
        raise FileNotFoundError(
            "pools_publico.json não encontrado na raiz do repositório")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["pools"]


def indice(matricula, tag, tamanho):
    h = hashlib.sha256(f"{tag}:{matricula}".encode()).hexdigest()
    return int(h[:12], 16) % tamanho


def atribuir(matricula, curso, pools=None):
    """Seed oficial da série do grupo. Determinístico e auditável."""
    if curso not in PERFIL:
        raise ValueError(f"curso deve ser um de {sorted(PERFIL)}")
    pools = pools or carregar_pools()
    p = pools[curso]
    return p[indice(matricula, f"demanda_{curso}", len(p))]


def caminho_serie(seed, curso, raiz=None):
    return os.path.join(raiz or RAIZ, "series", curso, f"serie_{seed:03d}.csv")
