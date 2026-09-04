
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "model" / "churn_model.joblib"
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Churn Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "age",
            "tenure_months",
            "monthly_charges",
            "support_calls",
            "contract_type"
        ]

        missing = [field for field in required_fields if field not in data]

        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        customer = pd.DataFrame([data])

        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])

        return jsonify({
            "prediction": prediction,
            "label": "Churn" if prediction == 1 else "Stay",
            "churn_probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
