import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Load the trained model
model = joblib.load("model_expanded.pkl")
all_training_cols = getattr(model, "feature_names_in_", None)

# Function to parse age range
def parse_age(age_str):
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2  # Return the midpoint of the age range
    except:
        return 30  # Fallback age

@app.route("/")
def home():
    return "Enhanced Substance Use Prediction API is running!"

@app.route("/predict_expanded", methods=["POST"])
def predict_expanded():
    """
    Expects JSON:
    {
      "Age": "30 to 34",
      "Gender": "Male",
      "Neighborhood": "downtown",
      "Substance": "fentanyl"
    }

    Returns:
    {
      "overdose_probability": 0.89,
      "overdose_class": 1,
      "confidence": "High",
      "explanation": "Substance suggests a high overdose risk."
    }
    """
    data = request.get_json() or {}

    # 1. Extract and process input data
    age_str = data.get("Age", "30 to 34")
    age_val = parse_age(age_str)

    gender_str = data.get("Gender", "unknown").strip().lower()
    gender_num = 1 if gender_str == "male" else 0  # Encode male as 1, others as 0

    neigh_str = data.get("Neighborhood", "unknown").strip().lower()
    subst_str = data.get("Substance", "none").strip().lower()

    # 2. Build input DataFrame
    df_input = pd.DataFrame({
        "AgeNumeric": [age_val],
        "GenderNum": [gender_num],
        "Neighborhood": [neigh_str],
        "Substance": [subst_str]
    })

    # 3. One-hot encode categorical features
    df_input_encoded = pd.get_dummies(df_input, columns=["Neighborhood", "Substance"],
                                      prefix=["neigh", "subst"])

    # 4. Align with model training columns
    if all_training_cols is not None:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0  # Fill missing columns with 0
        df_input_encoded = df_input_encoded[all_training_cols]  # Reorder columns

    # 5. Predict using the model
    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_proba = model.predict_proba(df_input_encoded)[0][1]  # Probability of overdose (class 1)
    overdose_probability = float(y_pred_proba)

    # 6. Assign confidence levels
    if overdose_probability > 0.7:
        confidence = "High"
    elif overdose_probability > 0.3:
        confidence = "Medium"
    else:
        confidence = "Low"

    # 7. Generate explanation
    explanation = []
    if "opioid" in subst_str or "fentanyl" in subst_str:
        explanation.append("Substance suggests a high overdose risk.")
    if neigh_str == "downtown":
        explanation.append("Downtown area might have higher incidents in this dataset.")
    if age_val < 25:
        explanation.append("Younger age group is correlated with certain substance use patterns (example).")

    explanation_str = " ".join(explanation) if explanation else "No specific high-risk factors detected in the input."

    # 8. Create and return JSON response
    response = {
        "overdose_probability": overdose_probability,
        "overdose_class": int(y_pred_class),
        "confidence": confidence,
        "explanation": explanation_str
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
