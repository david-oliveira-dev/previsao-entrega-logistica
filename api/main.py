"""API FastAPI que serve o modelo de tempo de entrega.

Carrega o Pipeline treinado (models/modelo.joblib) no startup e expõe:
  GET  /health   -> status do serviço e se o modelo carregou
  POST /predict  -> recebe as características do pedido e devolve dias previstos
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schema import PedidoEntrada, PrevisaoSaida

# Ordem exata das colunas que o pipeline espera (igual ao treino).
FEATURES = [
    "freight_value", "product_weight_g", "product_volume_cm3", "n_items",
    "same_state", "purchase_month", "purchase_dow", "customer_state", "seller_state",
]

MODELO_PATH = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"

app = FastAPI(
    title="Previsão de Tempo de Entrega — Olist",
    description="Estima o tempo de entrega (em dias) de um pedido de e-commerce.",
    version="0.1.0",
)

modelo = joblib.load(MODELO_PATH) if MODELO_PATH.exists() else None


def _montar_features(p: PedidoEntrada) -> pd.DataFrame:
    """Deriva as features calculadas e monta o DataFrame de uma linha."""
    d = date.fromisoformat(p.purchase_date)
    linha = {
        "freight_value": p.freight_value,
        "product_weight_g": p.product_weight_g,
        "product_volume_cm3": p.product_volume_cm3,
        "n_items": p.n_items,
        "same_state": int(p.customer_state.upper() == p.seller_state.upper()),
        "purchase_month": d.month,
        "purchase_dow": d.weekday(),
        "customer_state": p.customer_state.upper(),
        "seller_state": p.seller_state.upper(),
    }
    return pd.DataFrame([linha], columns=FEATURES)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "modelo_carregado": modelo is not None}


@app.post("/predict", response_model=PrevisaoSaida)
def predict(pedido: PedidoEntrada) -> PrevisaoSaida:
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado (models/modelo.joblib ausente).")
    previsao = modelo.predict(_montar_features(pedido))[0]
    return PrevisaoSaida(dias_previstos=round(float(previsao), 1))
