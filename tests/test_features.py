"""Testes da engenharia de features em nível de pedido (src/features.py)."""

from __future__ import annotations

import pandas as pd

from src.features import ALVO, FEATURES, construir_dataset


def _item(order_id, item_id, compra, entrega, **kw):
    """Cria uma linha nível-item com defaults sensatos, sobrescrevíveis por kw."""
    base = {
        "order_id": order_id,
        "order_item_id": item_id,
        "freight_value": 10.0,
        "product_weight_g": 500.0,
        "product_length_cm": 10.0,
        "product_height_cm": 10.0,
        "product_width_cm": 10.0,
        "customer_state": "SP",
        "seller_state": "SP",
        "order_purchase_timestamp": pd.Timestamp(compra),
        "order_delivered_customer_date": pd.Timestamp(entrega),
    }
    base.update(kw)
    return base


def test_agrega_itens_em_um_pedido():
    """Dois itens do mesmo pedido viram UMA linha, com somas corretas."""
    df = pd.DataFrame([
        _item("o1", 1, "2018-01-01", "2018-01-06", freight_value=10.0, product_weight_g=500.0),
        _item("o1", 2, "2018-01-01", "2018-01-06", freight_value=5.0, product_weight_g=300.0),
    ])
    ds = construir_dataset(df)

    assert len(ds) == 1
    linha = ds.iloc[0]
    assert linha["n_items"] == 2
    assert linha["freight_value"] == 15.0
    assert linha["product_weight_g"] == 800.0
    # volume = 10*10*10 por item, somado nos 2 itens
    assert linha["product_volume_cm3"] == 2000.0


def test_alvo_em_dias():
    """delivery_days é a diferença entrega - compra, em dias."""
    df = pd.DataFrame([_item("o1", 1, "2018-01-01", "2018-01-06")])
    ds = construir_dataset(df)
    assert ds.iloc[0][ALVO] == 5.0


def test_features_derivadas():
    """same_state, purchase_month e purchase_dow são derivadas da compra/UFs."""
    df = pd.DataFrame([
        _item("o1", 1, "2018-03-15", "2018-03-20", customer_state="SP", seller_state="MG"),
    ])
    ds = construir_dataset(df)
    linha = ds.iloc[0]
    assert linha["same_state"] == 0          # SP != MG
    assert linha["purchase_month"] == 3
    assert linha["purchase_dow"] == 3        # 2018-03-15 é quinta-feira


def test_same_state_quando_iguais():
    df = pd.DataFrame([_item("o1", 1, "2018-01-01", "2018-01-05", customer_state="RJ", seller_state="RJ")])
    assert construir_dataset(df).iloc[0]["same_state"] == 1


def test_remove_outliers_de_tempo():
    """Entregas <= 0 dia (erro) e >= 100 dias (cauda extrema) são descartadas."""
    df = pd.DataFrame([
        _item("ok", 1, "2018-01-01", "2018-01-06"),    # 5 dias  -> mantém
        _item("zero", 1, "2018-01-01", "2018-01-01"),  # 0 dias  -> remove
        _item("longo", 1, "2018-01-01", "2018-06-01"), # >100 d  -> remove
    ])
    ds = construir_dataset(df)
    assert list(ds["order_id"]) == ["ok"]


def test_descarta_linhas_com_nulos():
    """Linhas sem UF (ou outra feature obrigatória) são removidas."""
    df = pd.DataFrame([
        _item("o1", 1, "2018-01-01", "2018-01-06"),
        _item("o2", 1, "2018-01-01", "2018-01-06", customer_state=None),
    ])
    ds = construir_dataset(df)
    assert list(ds["order_id"]) == ["o1"]


def test_colunas_esperadas_presentes():
    """O dataset final expõe todas as FEATURES que o modelo consome + o alvo."""
    df = pd.DataFrame([_item("o1", 1, "2018-01-01", "2018-01-06")])
    ds = construir_dataset(df)
    for col in [*FEATURES, ALVO]:
        assert col in ds.columns
