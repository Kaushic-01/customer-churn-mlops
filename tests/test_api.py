
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import app


def test_home():
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200


def test_predict():
    client = app.test_client()

    payload = {
        "age": 35,
        "tenure_months": 8,
        "monthly_charges": 95.0,
        "support_calls": 5,
        "contract_type": "Monthly"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data
    assert "label" in data
    assert "churn_probability" in data


def test_missing_field():
    client = app.test_client()

    payload = {
        "age": 35
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
