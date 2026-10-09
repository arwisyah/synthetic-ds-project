"""Programmatically build ``analysis.ipynb`` using the ``nbformat`` library.

Run:
    python build_notebook.py

The generated notebook:
  (a) imports libraries and loads the ``.env`` variables,
  (b) connects to PostgreSQL and reads the ``customers`` table,
  (c) performs basic EDA including matplotlib visualizations,
  (d) trains a Random Forest model to predict ``target_churn``.
"""

from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT_FILE = "analysis.ipynb"

TITLE_CELL = """# Synthetic Customer Churn Analysis

This notebook loads the synthetic `customers` table from PostgreSQL and performs:

1. **Setup** — import libraries and load database credentials from `.env`.
2. **Data loading** — connect to PostgreSQL with SQLAlchemy and read the table.
3. **Exploratory Data Analysis (EDA)** — summary statistics and matplotlib visualizations.
4. **Modeling** — a Random Forest classifier predicting `target_churn`.
"""

IMPORT_CELL = """# (a) Import libraries and load .env variables
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load environment variables from the local .env file
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

print("Environment variables loaded.")"""

LOAD_CELL = """# (b) Connect to PostgreSQL and read the 'customers' table
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

df = pd.read_sql("SELECT * FROM customers", engine)

print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
df.head()"""

EDA_SHAPE_CELL = """# Schema, data types and non-null counts
df.info()"""

EDA_DESCRIBE_CELL = """# Summary statistics for the numeric columns
df.describe().T"""

EDA_MISSING_CELL = """# Missing values and target class balance
print("Missing values per column:")
print(df.isna().sum())

print("\\nTarget distribution (proportion):")
print(df["target_churn"].value_counts(normalize=True).rename("proportion"))"""

VIZ_HIST_CELL = """# (c) Basic EDA visualizations with matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(df["age"], bins=20, color="steelblue", edgecolor="black")
axes[0, 0].set_title("Age distribution")
axes[0, 0].set_xlabel("Age")

axes[0, 1].hist(df["income"], bins=20, color="seagreen", edgecolor="black")
axes[0, 1].set_title("Income distribution")
axes[0, 1].set_xlabel("Income")

axes[1, 0].hist(df["credit_score"], bins=20, color="darkorange", edgecolor="black")
axes[1, 0].set_title("Credit score distribution")
axes[1, 0].set_xlabel("Credit score")

df["target_churn"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1, 1], color=["#4c72b0", "#c44e52"], edgecolor="black"
)
axes[1, 1].set_title("Churn class balance")
axes[1, 1].set_xlabel("target_churn")
axes[1, 1].set_ylabel("count")

plt.tight_layout()
plt.show()"""

VIZ_CORR_CELL = """# Correlation heatmap (numeric columns, excluding the identifier)
numeric_df = df.select_dtypes(include=[np.number]).drop(
    columns=["customer_id"], errors="ignore"
)
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax)
ax.set_title("Correlation matrix")
plt.tight_layout()
plt.show()"""

MODEL_CELL = """# (d) Train a Random Forest classifier to predict target_churn
feature_cols = [c for c in df.columns if c not in ("target_churn", "customer_id")]

X = df[feature_cols]
y = df["target_churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n")
print(classification_report(y_test, y_pred, target_names=["no churn", "churn"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))"""

IMPORTANCE_CELL = """# Feature importances from the fitted Random Forest
importances = (
    pd.Series(model.feature_importances_, index=feature_cols)
    .sort_values(ascending=False)
)
print(importances)

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="teal", edgecolor="black")
ax.invert_yaxis()
ax.set_title("Random Forest feature importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()"""


def build_notebook() -> nbf.NotebookNode:
    cells = [
        new_markdown_cell(TITLE_CELL),
        new_markdown_cell("## 1. Setup"),
        new_code_cell(IMPORT_CELL),
        new_markdown_cell("## 2. Load data from PostgreSQL"),
        new_code_cell(LOAD_CELL),
        new_markdown_cell("## 3. Exploratory Data Analysis"),
        new_code_cell(EDA_SHAPE_CELL),
        new_code_cell(EDA_DESCRIBE_CELL),
        new_code_cell(EDA_MISSING_CELL),
        new_code_cell(VIZ_HIST_CELL),
        new_code_cell(VIZ_CORR_CELL),
        new_markdown_cell("## 4. Machine Learning: Random Forest"),
        new_code_cell(MODEL_CELL),
        new_code_cell(IMPORTANCE_CELL),
    ]

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.validate(notebook)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        nbf.write(notebook, fh)
    print(f"Wrote validated notebook with {len(notebook.cells)} cells to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
