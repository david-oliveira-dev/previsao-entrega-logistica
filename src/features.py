"""Engenharia de features (nível de PEDIDO) e construção do alvo.

Alvo: `delivery_days` = (entrega ao cliente − compra), em dias.
Um pedido pode ter vários itens; agregamos para uma linha por pedido com um
conjunto ENXUTO de features que a API consegue receber facilmente.
"""

from __future__ import annotations

import pandas as pd

# Features que o modelo (e a API) esperam.
NUMERICAS = ["freight_value", "product_weight_g", "product_volume_cm3", "n_items"]
CATEGORICAS = ["customer_state", "seller_state"]
DERIVADAS = ["same_state", "purchase_month", "purchase_dow"]
FEATURES = [*NUMERICAS, *DERIVADAS, *CATEGORICAS]
ALVO = "delivery_days"

# Limites para descartar outliers de tempo de entrega de forma justificada:
# entregas <= 0 dia são erro de registro; > 100 dias são casos raríssimos que
# distorcem o treino (caudas extremas, provável problema logístico atípico).
_MIN_DIAS, _MAX_DIAS = 0, 100


def construir_dataset(item_level: pd.DataFrame) -> pd.DataFrame:
    """Agrega os itens em nível de pedido e devolve features + alvo limpos."""
    df = item_level.copy()
    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )

    pedido = df.groupby("order_id").agg(
        freight_value=("freight_value", "sum"),
        product_weight_g=("product_weight_g", "sum"),
        product_volume_cm3=("product_volume_cm3", "sum"),
        n_items=("order_item_id", "count"),
        customer_state=("customer_state", "first"),
        seller_state=("seller_state", "first"),
        compra=("order_purchase_timestamp", "first"),
        entrega=("order_delivered_customer_date", "first"),
    ).reset_index()

    pedido[ALVO] = (pedido["entrega"] - pedido["compra"]).dt.total_seconds() / 86400
    pedido["same_state"] = (pedido["customer_state"] == pedido["seller_state"]).astype(int)
    pedido["purchase_month"] = pedido["compra"].dt.month
    pedido["purchase_dow"] = pedido["compra"].dt.dayofweek

    pedido = pedido.dropna(subset=[*NUMERICAS, *CATEGORICAS, ALVO])
    pedido = pedido[(pedido[ALVO] > _MIN_DIAS) & (pedido[ALVO] < _MAX_DIAS)]
    return pedido.reset_index(drop=True)
