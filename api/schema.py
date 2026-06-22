"""Contratos de entrada/saída da API (Pydantic).

A entrada usa campos "amigáveis" ao cliente; a API deriva sozinha as features
calculadas (mesma UF, mês e dia da semana da compra) antes de chamar o modelo.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PedidoEntrada(BaseModel):
    """Características de um pedido para prever o tempo de entrega."""

    freight_value: float = Field(..., ge=0, description="Valor do frete (R$)", examples=[15.1])
    product_weight_g: float = Field(..., ge=0, description="Peso total (g)", examples=[1200.0])
    product_volume_cm3: float = Field(..., ge=0, description="Volume total (cm³)", examples=[8000.0])
    n_items: int = Field(..., ge=1, description="Nº de itens no pedido", examples=[1])
    customer_state: str = Field(..., min_length=2, max_length=2, description="UF do cliente", examples=["SP"])
    seller_state: str = Field(..., min_length=2, max_length=2, description="UF do vendedor", examples=["MG"])
    purchase_date: str = Field(..., description="Data da compra (ISO, YYYY-MM-DD)", examples=["2018-03-15"])


class PrevisaoSaida(BaseModel):
    """Resposta da previsão."""

    dias_previstos: float = Field(..., description="Tempo de entrega estimado (dias)")
