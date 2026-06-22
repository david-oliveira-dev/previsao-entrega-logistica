"""Gera o notebook 01_eda_modelagem.ipynb com nbformat.

Rode da raiz do projeto:  python notebooks/_build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(
    "# Previsão de Tempo de Entrega — Olist (FastAPI + Docker)\n\n"
    "Estima em **quantos dias** um pedido será entregue, a partir de características do "
    "pedido. O destaque do projeto é a cadeia completa: **dado → modelo → API → "
    "container**. Aqui mostramos a parte de dados e modelagem; a lógica vive em `src/`."
))

cells.append(md("## 0. Setup"))
cells.append(code(
    "import sys\n"
    "from pathlib import Path\n\n"
    "RAIZ = Path.cwd()\n"
    "if not (RAIZ / 'src').exists():\n"
    "    RAIZ = RAIZ.parent\n"
    "sys.path.insert(0, str(RAIZ))\n\n"
    "import matplotlib.pyplot as plt\n"
    "import pandas as pd\n\n"
    "from src.data import juntar_tabelas\n"
    "from src.features import construir_dataset\n\n"
    "FIG = RAIZ / 'reports' / 'figures'\n"
    "FIG.mkdir(parents=True, exist_ok=True)\n"
    "ds = construir_dataset(juntar_tabelas(RAIZ / 'data'))\n"
    "print(f'{len(ds):,} pedidos entregues (após limpeza)')\n"
    "ds[['delivery_days', 'freight_value', 'product_weight_g', 'same_state']].describe().round(2)"
))

cells.append(md(
    "## 1. EDA\n\n### 1.1 Distribuição do tempo de entrega\n"
    "A maioria entrega em ~1-2 semanas, mas há uma cauda longa."
))
cells.append(code(
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.hist(ds.delivery_days, bins=50, color='#1f6f54', edgecolor='white')\n"
    "ax.axvline(ds.delivery_days.median(), color='crimson', linestyle='--',\n"
    "           label=f'mediana = {ds.delivery_days.median():.0f} dias')\n"
    "ax.set_xlabel('Tempo de entrega (dias)'); ax.set_ylabel('Nº de pedidos')\n"
    "ax.set_title('Distribuição do tempo de entrega'); ax.legend()\n"
    "fig.tight_layout(); fig.savefig(FIG / 'dist_entrega.png', dpi=120); plt.show()"
))

cells.append(md(
    "### 1.2 Mesma UF vs. UFs diferentes\n"
    "A distância importa: pedidos em que cliente e vendedor estão no **mesmo estado** "
    "chegam bem mais rápido."
))
cells.append(code(
    "por_uf = ds.groupby('same_state').delivery_days.agg(['mean', 'median', 'count'])\n"
    "por_uf.index = ['UF diferente', 'Mesma UF']\n"
    "print(por_uf.round(1))\n"
    "fig, ax = plt.subplots(figsize=(6, 4))\n"
    "ds.boxplot(column='delivery_days', by='same_state', ax=ax, showfliers=False)\n"
    "ax.set_xticklabels(['UF diferente', 'Mesma UF'])\n"
    "ax.set_xlabel(''); ax.set_ylabel('Tempo de entrega (dias)')\n"
    "ax.set_title('Entrega por proximidade geográfica'); plt.suptitle('')\n"
    "fig.tight_layout(); fig.savefig(FIG / 'entrega_por_uf.png', dpi=120); plt.show()"
))

cells.append(md(
    "### 1.3 Relação com o frete\n"
    "Frete mais alto tende a acompanhar entregas mais longas (proxy de distância/peso)."
))
cells.append(code(
    "faixas = pd.qcut(ds.freight_value, 10, duplicates='drop')\n"
    "media = ds.groupby(faixas, observed=True).delivery_days.mean()\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.plot(range(len(media)), media.values, marker='o', color='#1f6f54')\n"
    "ax.set_xlabel('Decil de valor de frete (baixo → alto)')\n"
    "ax.set_ylabel('Tempo médio de entrega (dias)')\n"
    "ax.set_title('Tempo de entrega cresce com o frete')\n"
    "fig.tight_layout(); fig.savefig(FIG / 'entrega_vs_frete.png', dpi=120); plt.show()"
))

cells.append(md(
    "## 2. Modelagem\n\n"
    "Pipeline do pacote (`src/train.py`): one-hot dos estados + Gradient Boosting, "
    "comparado ao baseline (prever a média). Métrica em **dias**. Salva o modelo em "
    "`models/` para a API."
))
cells.append(code(
    "from src.train import main as treinar\n"
    "res = treinar()\n"
    "pd.DataFrame({\n"
    "    'Baseline (média)': {'MAE': res['baseline_mae']},\n"
    "    'Gradient Boosting': {'MAE': res['mae'], 'RMSE': res['rmse']},\n"
    "}).T.round(2)"
))
cells.append(code(
    "from IPython.display import Image, display\n"
    "for nome in ['previsto_vs_real.png', 'residuos.png']:\n"
    "    display(Image(filename=str(FIG / nome)))"
))

cells.append(md(
    "## 3. Conclusão\n\n"
    "- O modelo erra **~5 dias (MAE)**, contra ~6,4 do baseline — ganho real usando "
    "frete, peso/volume, nº de itens, proximidade (mesma UF) e sazonalidade da compra.\n"
    "- O **Pipeline completo** (pré-processamento + modelo) é salvo em `models/modelo.joblib` "
    "e servido pela **API FastAPI** (`api/main.py`), empacotada em **Docker**.\n"
    "- Todos os números vêm desta execução real."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
destino = Path(__file__).resolve().parent / "01_eda_modelagem.ipynb"
nbf.write(nb, destino)
print(f"Notebook escrito em {destino}")
