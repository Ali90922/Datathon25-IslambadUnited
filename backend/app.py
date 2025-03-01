# app.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the saved model at startup
model = joblib.load("model.pkl")

def parse_age(age_str):
    """
    Same logic as in train_model.py (keep consistent).
    '30 to 34' -> 32
    """
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return 30  # fallback if parsing fails

@app.route("/", methods=["GET"])
def home():
    return "Substance Use Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON data, for example:
    {
      "Age": "30 to 34",
      "Gender": "Male"
    }
    Returns a JSON object with:
    {
      "isOpioidProbability": 0.73,
      "isOpioidClass": 1
    }
    """

    data = request.get_json()

    # Extract inputs
    age_str = data.get("Age", "30 to 34")
    gender_str = data.get("Gender", "Male")

    # Convert to numeric
    age_val = parse_age(age_str)
    gender_num = 1 if gender_str.strip().lower() == "male" else 0

    # Build a DataFrame for the model
    input_df = pd.DataFrame({
        "AgeNumeric": [age_val],
        "GenderNum": [gender_num]
    })

    # Predict class (0 or 1)
    y_pred_class = model.predict(input_df)[0]

    # Predict probability (for logistic regression, returns [prob_of_0, prob_of_1])
    y_pred_proba = model.predict_proba(input_df)[0][1]  # Probability of class "1"

    # Build response
    response = {
        "isOpioidProbability": float(y_pred_proba),
        "isOpioidClass": int(y_pred_class)
    }

    return jsonify(response)

if __name__ == "__main__":
    # Run Flask app (debug mode for development)
    app.run(debug=True, port=5000)

