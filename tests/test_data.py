"""Testes da junção das tabelas do Olist (src/data.py).

Geram CSVs minúsculos num diretório temporário e checam que o join devolve
só pedidos ENTREGUES com datas válidas.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import ARQUIVOS, juntar_tabelas


@pytest.fixture
def data_dir(tmp_path):
    """Cria as 5 tabelas mínimas do Olist em arquivos CSV temporários."""
    pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "delivered", "shipped"],   # o3 não entregue
        "order_purchase_timestamp": ["2018-01-01", "2018-01-02", "2018-01-03"],
        # o2 sem data de entrega -> deve ser descartado pelo dropna
        "order_delivered_customer_date": ["2018-01-06", None, "2018-01-09"],
    }).to_csv(tmp_path / ARQUIVOS["orders"], index=False)

    pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "order_item_id": [1, 1, 1],
        "product_id": ["p1", "p1", "p1"],
        "seller_id": ["s1", "s1", "s1"],
        "freight_value": [10.0, 12.0, 9.0],
    }).to_csv(tmp_path / ARQUIVOS["items"], index=False)

    pd.DataFrame({
        "product_id": ["p1"],
        "product_weight_g": [500.0],
        "product_length_cm": [10.0],
        "product_height_cm": [10.0],
        "product_width_cm": [10.0],
    }).to_csv(tmp_path / ARQUIVOS["products"], index=False)

    pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "customer_state": ["SP", "RJ", "MG"]}) \
        .to_csv(tmp_path / ARQUIVOS["customers"], index=False)
    pd.DataFrame({"seller_id": ["s1"], "seller_state": ["SP"]}) \
        .to_csv(tmp_path / ARQUIVOS["sellers"], index=False)

    return tmp_path


def test_mantem_apenas_entregues_com_datas(data_dir):
    """o2 (sem data de entrega) e o3 (não entregue) são filtrados; sobra o1."""
    df = juntar_tabelas(data_dir)
    assert list(df["order_id"]) == ["o1"]


def test_join_traz_colunas_das_5_tabelas(data_dir):
    """O resultado reúne colunas de orders, items, products, customers e sellers."""
    df = juntar_tabelas(data_dir)
    for col in ["freight_value", "product_weight_g", "customer_state", "seller_state"]:
        assert col in df.columns
    assert df.iloc[0]["customer_state"] == "SP"
    assert df.iloc[0]["seller_state"] == "SP"


def test_datas_viram_datetime(data_dir):
    """As colunas de data são parseadas como datetime (necessário para o alvo)."""
    df = juntar_tabelas(data_dir)
    assert pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(df["order_delivered_customer_date"])
