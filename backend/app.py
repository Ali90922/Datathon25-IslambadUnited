# app_expanded.py

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("model_expanded.pkl")

all_training_cols = getattr(model, "feature_names_in_", None)

def parse_age(age_str):
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return 30  # fallback

@app.route("/")
def home():
    return "Enhanced Substance Use Prediction API is running!"

@app.route("/predict_expanded", methods=["POST"])
def predict_expanded():
    """
    Expects JSON, e.g.:
    {
      "Age": "30 to 34",
      "Gender": "Male",
      "Neighborhood": "downtown",
      "Substance": "fentanyl"
      ... (optionally add date/time fields if your model used them)
    }

    Returns a richer JSON, for example:
    {
      "overdose_probability": 0.89,
      "overdose_class": 1,
      "confidence": "High",
      "explanation": "Model sees 'fentanyl' which strongly correlates with overdose incidents."
    }
    """
    data = request.get_json() or {}

    # 1. Extract and fallback
    age_str = data.get("Age", "30 to 34")
    age_val = parse_age(age_str)

    gender_str = data.get("Gender", "unknown").strip().lower()
    gender_num = 1 if gender_str == "male" else 0

    neigh_str = data.get("Neighborhood", "unknown").strip().lower()
    subst_str = data.get("Substance", "none").strip().lower()

    # 2. Build input DataFrame
    df_input = pd.DataFrame({
        "AgeNumeric": [age_val],
        "GenderNum": [gender_num],
        "Neighborhood": [neigh_str],
        "Substance": [subst_str]
    })

    # (Optional) If your model uses date/time features, parse them here
    # e.g. data.get("DispatchDate") -> day_of_week, hour, etc.

    # 3. One-hot encode
    df_input_encoded = pd.get_dummies(df_input, columns=["Neighborhood", "Substance"],
                                      prefix=["neigh", "subst"])

    # 4. Align with training columns
    if all_training_cols is not None:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[all_training_cols]

    # 5. Predict
    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_proba = model.predict_proba(df_input_encoded)[0][1]  # Probability of class=1
    overdose_probability = float(y_pred_proba)

    # 6. Provide a "confidence" descriptor
    #    e.g. Low (< 0.3), Medium (0.3 to 0.7), High (> 0.7)
    if overdose_probability > 0.7:
        confidence = "High"
    elif overdose_probability > 0.3:
        confidence = "Medium"
    else:
        confidence = "Low"

    # 7. Optional: Simple heuristic explanation
    explanation = []
    if "opioid" in subst_str or "fentanyl" in subst_str:
        explanation.append("Substance suggests a high overdose risk.")
    if neigh_str == "downtown":
        explanation.append("Downtown area might have higher incidents in this dataset.")
    if age_val < 25:
        explanation.append("Younger age group is correlated with certain substance use patterns (example).")

    if not explanation:
        explanation_str = "No specific high-risk factors detected in the input."
    else:
        explanation_str = " ".join(explanation)

    # 8. Build response
    response = {
        "overdose_probability": overdose_probability,
        "overdose_class": int(y_pred_class),
        "confidence": confidence,
        "explanation": explanation_str
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

