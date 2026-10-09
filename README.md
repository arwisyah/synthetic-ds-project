# Customer Churn Prediction from Synthetic Data (PostgreSQL + Random Forest)

An end-to-end, reproducible pipeline that **generates synthetic customer
records into a PostgreSQL database** and **predicts customer churn** with a
Random Forest classifier. It covers the full workflow in one place: synthetic
data generation, database ingestion, exploratory data analysis (EDA) in a
Jupyter notebook, and model training and evaluation.

> ## ⚠️ AI-Generated Disclaimer
> This entire project — the code, the synthetic data, the Jupyter notebook, and
> this documentation — was **generated end-to-end by an AI agent** (OpenCode,
> using the **Big Pickle** model). No part of it was hand-authored by a human.
> It is intended for demonstration and educational purposes and should be
> reviewed, validated, and adapted before any production or research use.

## Project structure

```
.
├── .env                 # database credentials (git-ignored)
├── .gitignore
├── docker-compose.yml   # optional local PostgreSQL (container)
├── requirements.txt
├── generate_data.py     # generates 1,000 x 10 synthetic rows -> 'customers'
├── build_notebook.py    # builds analysis.ipynb with nbformat
├── analysis.ipynb       # EDA + Random Forest model
└── README.md
```

## 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 2. Database

Two options:

- **Existing local PostgreSQL** (what `.env` points to by default):

  ```
  POSTGRES_USER=postgres
  POSTGRES_PASSWORD=learning
  POSTGRES_DB=postgres
  POSTGRES_HOST=localhost
  POSTGRES_PORT=5432
  ```

- **Container** (optional): `docker compose up -d`. The container publishes host
  port **5433**, so set `POSTGRES_PORT=5433` in `.env` if you use it.

## 3. Generate the data

```bash
python generate_data.py
```

Creates/replaces the `customers` table with 1,000 rows and 10 columns.

## 4. Build and run the notebook

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

Or simply open `analysis.ipynb` in Jupyter and run all cells.
