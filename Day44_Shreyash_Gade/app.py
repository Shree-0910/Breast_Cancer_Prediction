from flask import Flask, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load(open("model.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return "Breast Cancer Prediction API is Running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Check if JSON is provided
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        features = data.get("features")

        # Validate number of features
        if len(features) != 30:
            return jsonify({"error": "Model expects 30 features"}), 400

        # Convert input to numpy
        input_data = np.array(features).reshape(1, -1)

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        result = "Malignant (Cancer)" if prediction == 0 else "Benign (No Cancer)"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
