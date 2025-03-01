import joblib
import pandas as pd
import re

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# --------------------------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------------------------
model = joblib.load("model_expanded.pkl")
all_training_cols = getattr(model, "feature_names_in_", None)

# --------------------------------------------------------------------
# 2. Helper function to convert range string to midpoint integer
# --------------------------------------------------------------------
def parse_age(age_str):
    """
    If the user or code supplies an Age string like '30 to 34',
    this converts it to the midpoint: (30+34)//2 = 32.
    Fallback is 30 if anything fails.
    """
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return 30  # fallback

# --------------------------------------------------------------------
# 3. The main prediction logic
#    (Identical to your existing /predict_expanded code)
# --------------------------------------------------------------------
def make_prediction(age_str, gender_str, neigh_str, subst_str):
    # 1. Convert age range to midpoint
    age_val = parse_age(age_str)

    # 2. Encode gender
    gender_num = 1 if gender_str.strip().lower() == "male" else 0

    # 3. Build input DataFrame
    df_input = pd.DataFrame({
        "AgeNumeric": [age_val],
        "GenderNum": [gender_num],
        "Neighborhood": [neigh_str.strip().lower()],
        "Substance": [subst_str.strip().lower()]
    })

    # 4. One-hot encode
    df_input_encoded = pd.get_dummies(df_input, columns=["Neighborhood", "Substance"],
                                      prefix=["neigh", "subst"])

    # 5. Align columns with training
    if all_training_cols is not None:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[all_training_cols]

    # 6. Predict
    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_proba = model.predict_proba(df_input_encoded)[0][1]
    overdose_probability = float(y_pred_proba)

    # 7. Confidence
    if overdose_probability > 0.7:
        confidence = "High"
    elif overdose_probability > 0.3:
        confidence = "Medium"
    else:
        confidence = "Low"

    # 8. Simple explanation logic
    explanation = []
    if "opioid" in subst_str.lower() or "fentanyl" in subst_str.lower():
        explanation.append("Substance suggests a high overdose risk.")
    if neigh_str.lower() == "downtown":
        explanation.append("Downtown area might have higher incidents in this dataset.")
    # If you want to parse numeric from the age_str, do it now:
    #   e.g., parse_age("30 to 34") -> 32, then check < 25
    midpoint_age = parse_age(age_str)
    if midpoint_age < 25:
        explanation.append("Younger age group is correlated with certain substance use patterns (example).")

    explanation_str = " ".join(explanation) if explanation else "No specific high-risk factors detected in the input."

    return {
        "overdose_probability": overdose_probability,
        "overdose_class": int(y_pred_class),
        "confidence": confidence,
        "explanation": explanation_str
    }

# --------------------------------------------------------------------
# 4. Flask endpoints
# --------------------------------------------------------------------
@app.route("/")
def home():
    return "Enhanced Substance Use Prediction API is running!"

@app.route("/predict_expanded", methods=["POST"])
def predict_expanded():
    """
    Same as before: expects JSON with Age, Gender, Neighborhood, Substance
    """
    data = request.get_json() or {}

    age_str = data.get("Age", "30 to 34")
    gender_str = data.get("Gender", "unknown")
    neigh_str = data.get("Neighborhood", "unknown")
    subst_str = data.get("Substance", "none")

    result = make_prediction(age_str, gender_str, neigh_str, subst_str)
    return jsonify(result)

# --------------------------------------------------------------------
# 5. NEW endpoint: /predict_from_text
#    Takes a free-text query and tries to parse Age, Gender, Neighborhood, Substance
# --------------------------------------------------------------------
@app.route("/predict_from_text", methods=["POST"])
def predict_from_text():
    """
    Expects JSON like:
        {
           "text": "What is the substance use of a 35 years old female in Robertson?"
        }
    Parses out:
      - Age (from "35 years old")
      - Gender ("female" or "male")
      - Neighborhood (simple approach: we'll guess 'Robertson' if we see that word)
      - Substance: we'll guess from a small list, or fallback to "none"
    Then calls make_prediction(...) and returns the same structure.
    """
    body = request.get_json() or {}
    user_text = body.get("text", "").lower()

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # A) Parse Age from "NN years old"
    age_str_detected = None
    match = re.search(r"(\d{1,3})\s?years?\s?old", user_text)
    if match:
        # If found, convert '35' to a range string "35 to 35"
        numeric_age = match.group(1)
        age_str_detected = f"{numeric_age} to {numeric_age}"
    else:
        # Fallback if we can't find numeric age
        age_str_detected = "30 to 34"  # or any default

    # B) Parse Gender
    gender_detected = "unknown"
    if "male" in user_text:
        gender_detected = "male"
    elif "female" in user_text:
        gender_detected = "female"

    # C) Parse Neighborhood
    #   For simplicity, let's assume the user might mention "robertson" directly.
    #   We'll do a naive approach: find any capitalized word or something custom.
    #   But let's do it with a simple guess:
    neighborhood_detected = "unknown"
    # If you know there's a set of known neighborhoods, you can do partial matching.
    # For demonstration, we do naive:
    # Check each word, if it looks "robertson," or "downtown," etc.
    # We'll just do a mini-check:
    possible_neighborhoods = ["robertson", "downtown", "brooklyn", "queens"]  # example list
    for neigh in possible_neighborhoods:
        if neigh in user_text:
            neighborhood_detected = neigh
            break

    # D) Parse Substance
    #   We'll do the same naive approach for known substances:
    substance_detected = "none"
    possible_substances = ["fentanyl", "opioid", "heroin", "alcohol", "cocaine", "meth"]
    for s in possible_substances:
        if s in user_text:
            substance_detected = s
            break

    # E) Now we have age_str_detected, gender_detected, neighborhood_detected, substance_detected
    #    Let's call the same logic used in /predict_expanded
    result = make_prediction(
        age_str_detected,
        gender_detected,
        neighborhood_detected,
        substance_detected
    )

    # Return JSON
    parsed_data = {
        "Age": age_str_detected,
        "Gender": gender_detected,
        "Neighborhood": neighborhood_detected,
        "Substance": substance_detected
    }
    return jsonify({
        "parsed_data": parsed_data,
        "prediction": result
    })

# --------------------------------------------------------------------
# 6. Run the Flask server
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
