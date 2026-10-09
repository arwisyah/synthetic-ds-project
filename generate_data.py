"""Generate a synthetic customer dataset and load it into PostgreSQL.

The script produces a tabular dataset with exactly 10 columns and 1,000 rows,
then writes it to the ``customers`` table using SQLAlchemy. Database
credentials are read from a local ``.env`` file.

Run:
    python generate_data.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

N_ROWS = 1000
RANDOM_SEED = 42
TABLE_NAME = "customers"


def get_engine():
    """Build a SQLAlchemy engine from the credentials stored in ``.env``."""
    load_dotenv()

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_DB", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)


def generate_customers(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a synthetic customer DataFrame with ``n_rows`` rows and 10 columns."""
    rng = np.random.default_rng(seed)

    customer_id = np.arange(1, n_rows + 1)
    age = np.clip(rng.normal(42, 12, n_rows), 18, 75).round().astype(int)
    income = np.clip(rng.normal(60_000, 25_000, n_rows), 15_000, None).round(2)
    # Estimated salary is correlated with income (plus age effect and noise).
    estimated_salary = np.clip(
        income * 0.9 + age * 400 + rng.normal(0, 8_000, n_rows), 15_000, None
    ).round(2)
    credit_score = np.clip(rng.normal(650, 80, n_rows), 300, 850).round().astype(int)
    account_balance = np.clip(rng.normal(55_000, 40_000, n_rows), 0, None).round(2)
    tenure_months = rng.integers(1, 121, n_rows)
    num_products = rng.integers(1, 5, n_rows)
    is_active = rng.integers(0, 2, n_rows)

    # Latent churn propensity modelled as a logistic function of the observable
    # features, so that ``target_churn`` is genuinely learnable rather than noise.
    z = (
        0.9 * (is_active == 0)
        - 0.00002 * (income - 60_000)
        - 0.006 * (credit_score - 650)
        - 0.35 * (num_products - 2)
        + 0.40 * (age - 42) / 12
        - 0.25 * (account_balance - 55_000) / 40_000
        + 0.15
    )
    churn_prob = 1.0 / (1.0 + np.exp(-z))
    target_churn = (rng.random(n_rows) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "account_balance": account_balance,
            "tenure_months": tenure_months,
            "num_products": num_products,
            "is_active": is_active,
            "estimated_salary": estimated_salary,
            "target_churn": target_churn,
        }
    )


def main() -> None:
    df = generate_customers()
    assert df.shape == (N_ROWS, 10), f"Unexpected shape: {df.shape}"

    engine = get_engine()
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print(f"Generated dataset with shape {df.shape}")
    print(f"Wrote {len(df)} rows to table '{TABLE_NAME}'")
    print(f"Churn rate: {df['target_churn'].mean():.3%}")
    print(df.head())


if __name__ == "__main__":
    main()
