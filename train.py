
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def create_dataset(n=1200, random_state=42):
    np.random.seed(random_state)

    age = np.random.randint(18, 75, n)
    tenure_months = np.random.randint(1, 73, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    support_calls = np.random.poisson(2, n)

    contract_type = np.random.choice(
        ["Monthly", "One Year", "Two Year"],
        size=n,
        p=[0.55, 0.25, 0.20]
    )

    risk = (
        -0.03 * tenure_months
        + 0.025 * monthly_charges
        + 0.35 * support_calls
        + np.where(contract_type == "Monthly", 1.1, 0)
        + np.where(contract_type == "Two Year", -1.1, 0)
        - 2.0
    )

    probability = 1 / (1 + np.exp(-risk))
    churn = np.random.binomial(1, probability)

    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "contract_type": contract_type,
        "churn": churn
    })


def train_model():
    df = create_dataset()

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_calls"
    ]

    categorical_features = ["contract_type"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)

    model_dir = Path(__file__).parent / "model"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "churn_model.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Model accuracy: {score:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()
