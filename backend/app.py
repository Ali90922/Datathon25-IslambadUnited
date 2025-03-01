# app_expanded.py

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the expanded model
model = joblib.load("model_expanded.pkl")

# If your scikit-learn version supports it, we can retrieve the training columns:
all_training_cols = getattr(model, "feature_names_in_", None)

def parse_age(age_str):
    """
    Convert '30 to 34' -> 32. If parsing fails, default to 30.
    """
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return 30  # fallback if user input is malformed

@app.route("/")
def home():
    return "Expanded Substance Use Prediction API is running!"

@app.route("/predict_expanded", methods=["POST"])
def predict_expanded():
    """
    Expects JSON data, e.g.:
    {
      "Age": "30 to 34",
      "Gender": "Male",
      "Neighborhood": "riverheights",
      "Substance": "opioids"
    }
    Some fields can be missing. We'll fallback to defaults:
      - Age -> "30 to 34"
      - Gender -> "unknown"
      - Neighborhood -> "unknown"
      - Substance -> "none"

    Returns JSON:
    {
      "overdose_probability": float,
      "overdose_class": int
    }
    """
    data = request.get_json() or {}

    # 1. Extract user inputs, fallback if missing
    age_str = data.get("Age", "30 to 34")
    age_val = parse_age(age_str)

    gender_str = data.get("Gender", "unknown").strip().lower()
    gender_num = 1 if gender_str == "male" else 0

    neigh_str = data.get("Neighborhood", "unknown").strip().lower()
    subst_str = data.get("Substance", "none").strip().lower()

    # 2. Build a mini DataFrame with the user input
    input_dict = {
        "AgeNumeric": [age_val],
        "GenderNum": [gender_num],
        "Neighborhood": [neigh_str],
        "Substance": [subst_str]
    }
    df_input = pd.DataFrame(input_dict)

    # 3. One-hot encode Neighborhood & Substance
    df_input_encoded = pd.get_dummies(
        df_input,
        columns=["Neighborhood", "Substance"],
        prefix=["neigh", "subst"]
    )

    # 4. Align with training columns
    #    If we have the feature list from model.feature_names_in_, we can fill missing columns with 0
    if all_training_cols is not None:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0

        # Also ensure the same column order
        df_input_encoded = df_input_encoded[all_training_cols]
    else:
        # If scikit-learn version doesn't have feature_names_in_, 
        # you'd store them manually or skip the ordering step
        pass

    # 5. Predict with the loaded RandomForest
    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_probs = model.predict_proba(df_input_encoded)[0]
    # Probability of class=1 (overdose)
    overdose_prob = float(y_pred_probs[1])

    response = {
        "overdose_probability": overdose_prob,
        "overdose_class": int(y_pred_class)
    }

    return jsonify(response)

if __name__ == "__main__":
    # Run Flask with debug mode
    app.run(debug=True, port=5000)

