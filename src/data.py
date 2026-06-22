"""Carregamento e junção das tabelas do Olist (nível de item de pedido).

Mantém apenas pedidos **entregues** com datas de compra e entrega válidas — só
neles o tempo de entrega real é conhecido. As tabelas são unidas pelas chaves:
orders → items (order_id) → products (product_id) / customers (customer_id) /
sellers (seller_id).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARQUIVOS = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}

_DATAS = ["order_purchase_timestamp", "order_delivered_customer_date"]


def juntar_tabelas(data_dir: str | Path) -> pd.DataFrame:
    """Une as 5 tabelas num DataFrame nível-item de pedidos entregues válidos."""
    d = Path(data_dir)

    orders = pd.read_csv(d / ARQUIVOS["orders"], parse_dates=_DATAS)
    orders = orders[orders["order_status"] == "delivered"].dropna(subset=_DATAS)

    items = pd.read_csv(d / ARQUIVOS["items"])
    products = pd.read_csv(
        d / ARQUIVOS["products"],
        usecols=["product_id", "product_weight_g",
                 "product_length_cm", "product_height_cm", "product_width_cm"],
    )
    customers = pd.read_csv(d / ARQUIVOS["customers"], usecols=["customer_id", "customer_state"])
    sellers = pd.read_csv(d / ARQUIVOS["sellers"], usecols=["seller_id", "seller_state"])

    return (
        orders
        .merge(items, on="order_id")
        .merge(products, on="product_id", how="left")
        .merge(customers, on="customer_id", how="left")
        .merge(sellers, on="seller_id", how="left")
    )
