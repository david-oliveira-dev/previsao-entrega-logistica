"""Testes da API FastAPI (api/main.py) e dos contratos Pydantic (api/schema.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import _montar_features, app

client = TestClient(app)

PEDIDO_VALIDO = {
    "freight_value": 15.1,
    "product_weight_g": 1200.0,
    "product_volume_cm3": 8000.0,
    "n_items": 1,
    "customer_state": "SP",
    "seller_state": "MG",
    "purchase_date": "2018-03-15",
}


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    # O modelo versionado em models/ deve carregar no startup.
    assert r.json()["modelo_carregado"] is True


def test_predict_retorna_numero_positivo():
    r = client.post("/predict", json=PEDIDO_VALIDO)
    assert r.status_code == 200
    dias = r.json()["dias_previstos"]
    assert isinstance(dias, float)
    assert dias > 0


@pytest.mark.parametrize("campo,valor", [
    ("n_items", 0),            # ge=1
    ("freight_value", -1.0),   # ge=0
    ("customer_state", "São"), # max_length=2
])
def test_predict_rejeita_entrada_invalida(campo, valor):
    payload = {**PEDIDO_VALIDO, campo: valor}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_montar_features_deriva_corretamente():
    """same_state, mês e dia-da-semana são derivados; UFs viram maiúsculas."""
    from api.schema import PedidoEntrada

    p = PedidoEntrada(**{**PEDIDO_VALIDO, "customer_state": "sp", "seller_state": "sp"})
    linha = _montar_features(p).iloc[0]
    assert linha["same_state"] == 1          # sp == sp
    assert linha["customer_state"] == "SP"   # maiúsculas
    assert linha["purchase_month"] == 3
    assert linha["purchase_dow"] == 3        # 2018-03-15 é quinta-feira


def test_montar_features_ordem_das_colunas():
    """O DataFrame sai na ordem EXATA que o pipeline treinado espera."""
    from api.main import FEATURES
    from api.schema import PedidoEntrada

    p = PedidoEntrada(**PEDIDO_VALIDO)
    assert list(_montar_features(p).columns) == FEATURES
