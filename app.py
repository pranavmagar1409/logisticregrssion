import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# Load the trained Logistic Regression model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Exact feature names/order the model was trained with
FEATURES = list(model.feature_names_in_)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Logistic Regression Predictor</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 560px; margin: 2rem auto; padding: 0 1rem; }
    label { display: block; margin-top: .8rem; font-size: .9rem; }
    input { width: 100%; padding: .5rem; box-sizing: border-box; }
    button { margin-top: 1.2rem; padding: .7rem 1.2rem; cursor: pointer; }
    .result { margin-top: 1.5rem; padding: 1rem; background: #eef6ff; border-radius: 8px; }
    .error { background: #ffecec; }
  </style>
</head>
<body>
  <h1>Logistic Regression Predictor</h1>
  <form method="post" action="/">
    {% for f in features %}
      <label>{{ f.replace('_', ' ') }}
        <input type="number" step="any" name="{{ f }}" value="{{ values.get(f, '') }}" required>
      </label>
    {% endfor %}
    <button type="submit">Predict</button>
  </form>

  {% if error %}
    <div class="result error">{{ error }}</div>
  {% elif prediction %}
    <div class="result">
      <strong>Prediction: {{ prediction }}</strong>
      <ul>
        {% for cls, p in probabilities.items() %}
          <li>{{ cls }}: {{ "%.1f"|format(p * 100) }}%</li>
        {% endfor %}
      </ul>
    </div>
  {% endif %}
</body>
</html>
"""


def run_prediction(data: dict):
    """Build a one-row DataFrame in the training column order and predict."""
    row = {f: float(data[f]) for f in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)
    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]
    return str(pred), {str(c): float(p) for c, p in zip(model.classes_, probs)}


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {"features": FEATURES, "values": {}, "prediction": None,
           "probabilities": {}, "error": None}
    if request.method == "POST":
        ctx["values"] = request.form.to_dict()
        try:
            ctx["prediction"], ctx["probabilities"] = run_prediction(request.form)
        except (KeyError, ValueError) as e:
            ctx["error"] = f"Invalid input: {e}"
    return render_template_string(HTML, **ctx)


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API. Example body: {"Pre_Semester_GPA": 3.2, ...all 11 features...}"""
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400
    try:
        pred, probs = run_prediction(data)
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    return jsonify({"prediction": pred, "probabilities": probs})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": FEATURES})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
