from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate: features key must exist
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request body."}), 400

    features = data["features"]

    # Validate: exactly 4 values
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Field 'features' must be a list of exactly 4 numeric values."}), 400

    # Validate: all values are numeric
    for val in features:
        if not isinstance(val, (int, float)):
            return jsonify({"error": f"All features must be numeric. Got: {val!r}"}), 400

    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return jsonify({
        "predicted_class": target_names[prediction],
        "probabilities": {name: round(float(prob), 4) for name, prob in zip(target_names, proba)}
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)

    if "samples" not in data:
        return jsonify({"error": "Missing 'samples' key in request body."}), 400

    samples = data["samples"]

    if not isinstance(samples, list) or len(samples) == 0:
        return jsonify({"error": "Field 'samples' must be a non-empty list of feature arrays."}), 400

    results = []
    for i, features in enumerate(samples):
        if not isinstance(features, list) or len(features) != 4:
            return jsonify({"error": f"Sample {i}: must be a list of exactly 4 numeric values."}), 400
        for val in features:
            if not isinstance(val, (int, float)):
                return jsonify({"error": f"Sample {i}: all features must be numeric. Got: {val!r}"}), 400

        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        results.append({
            "predicted_class": target_names[prediction],
            "probabilities": {name: round(float(prob), 4) for name, prob in zip(target_names, proba)}
        })

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
