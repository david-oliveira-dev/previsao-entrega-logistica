"""Treina o modelo de tempo de entrega e salva o Pipeline COMPLETO em models/.

O Pipeline embute o pré-processamento (one-hot dos estados + numéricas), então
a MESMA transformação é aplicada na inferência da API. Métrica em DIAS.

Uso:  python -m src.train
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import juntar_tabelas
from .features import ALVO, CATEGORICAS, DERIVADAS, FEATURES, NUMERICAS, construir_dataset

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
MODELS = RAIZ / "models"
FIGURAS = RAIZ / "reports" / "figures"


def construir_pipeline() -> Pipeline:
    """One-hot nos estados, numéricas/derivadas passam direto, e o regressor."""
    numericas = [*NUMERICAS, *DERIVADAS]
    pre = ColumnTransformer([
        ("estados", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("num", "passthrough", numericas),
    ])
    # Gradient Boosting: modelo aditivo e COMPACTO (poucos MB) — cabe no repositório
    # versionado, requisito para a API funcionar para quem clonar.
    modelo = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.9, random_state=42
    )
    return Pipeline([("pre", pre), ("modelo", modelo)])


def _figuras(y_real: np.ndarray, y_prev: np.ndarray) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    # Previsto vs. real
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_real, y_prev, alpha=0.2, edgecolor="none", color="#1f6f54")
    lim = np.percentile(y_real, 99)
    ax.plot([0, lim], [0, lim], "--", color="crimson", label="previsão perfeita")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Entrega real (dias)"); ax.set_ylabel("Entrega prevista (dias)")
    ax.set_title("Previsto vs. Real"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURAS / "previsto_vs_real.png", dpi=120); plt.close(fig)
    # Resíduos
    res = y_real - y_prev
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res, bins=40, color="#1f6f54", edgecolor="white")
    ax.axvline(0, color="crimson", linestyle="--")
    ax.set_xlabel("Resíduo = real − previsto (dias)"); ax.set_ylabel("Frequência")
    ax.set_title("Distribuição dos resíduos")
    fig.tight_layout(); fig.savefig(FIGURAS / "residuos.png", dpi=120); plt.close(fig)


def main() -> dict[str, float]:
    ds = construir_dataset(juntar_tabelas(DATA))
    print(f"Pedidos após limpeza: {len(ds)}")
    X, y = ds[FEATURES], ds[ALVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline: prever sempre a média do treino.
    media = y_tr.mean()
    base_mae = mean_absolute_error(y_te, np.full(len(y_te), media))
    base_rmse = root_mean_squared_error(y_te, np.full(len(y_te), media))
    print(f"Baseline (média={media:.1f}d)   MAE={base_mae:.2f}  RMSE={base_rmse:.2f}  (dias)")

    pipe = construir_pipeline()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = root_mean_squared_error(y_te, pred)
    print(f"Gradient Boosting           MAE={mae:.2f}  RMSE={rmse:.2f}  (dias)")

    _figuras(y_te.to_numpy(), pred)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS / "modelo.joblib")
    print(f"Modelo salvo em {MODELS / 'modelo.joblib'}")

    return {"baseline_mae": base_mae, "mae": mae, "rmse": rmse, "media_treino": media}


if __name__ == "__main__":
    main()
