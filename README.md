
# Customer Churn Prediction API

A beginner-friendly MLOps project that demonstrates the journey from
machine learning model training to API serving, testing, Git version control,
and deployment-ready project organization.

## Project workflow

Data -> Train Model -> Save Model -> Flask API -> Test -> Git -> GitHub

## Setup

```bash
pip install -r requirements.txt
```

## Train model

```bash
python train.py
```

## Run API

```bash
python app.py
```

## Run tests

```bash
pytest
```

## Prediction endpoint

POST `/predict`

Example JSON:

```json
{
  "age": 35,
  "tenure_months": 8,
  "monthly_charges": 95,
  "support_calls": 5,
  "contract_type": "Monthly"
}
```
