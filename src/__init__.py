"""Previsão de tempo de entrega (Olist) — pipeline de dados, treino e API.

Fluxo: data (junção das tabelas) -> features (alvo + engenharia) -> train
(pipeline sklearn salvo em models/) -> api (FastAPI serve o modelo).
"""

__version__ = "0.1.0"
