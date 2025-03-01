import joblib
import pandas as pd
import re

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# ------------------------------------------------------------------
# 1. Load Model
# ------------------------------------------------------------------
model = joblib.load("model_expanded.pkl")
all_training_cols = getattr(model, "feature_names_in_", None)

# ------------------------------------------------------------------
# 2. Load Dataset
#    (Use your real CSV name; e.g., 'Substance_Use_20250301.csv')
# ------------------------------------------------------------------
df_dataset = pd.read_csv("Substance_Use_20250301.csv")  # or "dataset.csv" if renamed
df_dataset["Neighbourhood"] = df_dataset["Neighbourhood"].astype(str).str.lower()
df_dataset["Substance"] = df_dataset["Substance"].astype(str).str.lower()
df_dataset["Age"] = df_dataset["Age"].astype(str)

valid_neighborhoods = set(df_dataset["Neighbourhood"].unique())
valid_substances = set(df_dataset["Substance"].unique())
valid_age_ranges = set(df_dataset["Age"].unique())  # e.g. "20 to 24", "25 to 29", etc.

# ------------------------------------------------------------------
# 3. Precompute all numeric ranges for easy lookup
# ------------------------------------------------------------------
age_bins = []
for age_range_str in valid_age_ranges:
    parts = age_range_str.split(" to ")
    if len(parts) == 2:
        try:
            low = int(parts[0])
            high = int(parts[1])
            age_bins.append((low, high, age_range_str))
        except ValueError:
            # Not a valid "X to Y" format
            pass

def map_numeric_age_to_range(numeric_age):
    """
    Given a numeric age (e.g., 23), find which "X to Y" range it falls into.
    Returns the matching string, e.g. "20 to 24" or None if none found.
    """
    for (low, high, label) in age_bins:
        if low <= numeric_age <= high:
            return label
    return None

# ------------------------------------------------------------------
# 4. Helper Functions for Model
# ------------------------------------------------------------------
def parse_age(age_str):
    """Convert '30 to 34' into an integer midpoint 32, etc. Fallback=30 if fail."""
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return 30

def build_input_dataframe(age_val, gender_str, neigh_str, subst_str):
    """Build and one-hot encode the input row to match the model's expected columns."""
    # Encode gender: 1 for 'male', else 0
    gender_num = 1 if gender_str.lower() == "male" else 0

    # Prepare input DataFrame
    df_input = pd.DataFrame({
        "AgeNumeric": [age_val],
        "GenderNum": [gender_num],
        "Neighborhood": [neigh_str.lower()],
        "Substance": [subst_str.lower()]
    })

    # One-hot encode
    df_input_encoded = pd.get_dummies(df_input, columns=["Neighborhood", "Substance"],
                                      prefix=["neigh", "subst"])
    # Align with training columns
    if all_training_cols:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[all_training_cols]

    return df_input_encoded

def generate_explanation(prob, age_val, gender_str, neigh_str, subst_str):
    explanation = []
    # Example logic
    if "opioid" in subst_str or "fentanyl" in subst_str:
        explanation.append("Substance suggests a higher overdose risk.")
    if neigh_str == "downtown":
        explanation.append("Downtown area might have higher incidents in the dataset.")
    if age_val < 25:
        explanation.append("Younger age group might have certain higher risk factors.")

    if not explanation:
        return "No specific high-risk factors detected in the input."
    return " ".join(explanation)

def assign_confidence(prob):
    if prob > 0.7:
        return "High"
    elif prob > 0.3:
        return "Medium"
    else:
        return "Low"

# ------------------------------------------------------------------
# 5. Simple partial matching function for Neighborhood/Substance
# ------------------------------------------------------------------
def partial_match(term, candidates):
    """
    Return the first candidate from 'candidates' that is a substring of 'term'
    OR 'term' is a substring of that candidate. If none found, return None.
    """
    term_lower = term.lower()
    for c in candidates:
        c_lower = c.lower()
        # Simple check: either "downtown" in "downtown area" or vice versa
        if term_lower in c_lower or c_lower in term_lower:
            return c
    return None

# ------------------------------------------------------------------
# 6. Flask Endpoints
# ------------------------------------------------------------------
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
    """
    data = request.get_json() or {}
    age_str = data.get("Age", "30 to 34")
    gender_str = data.get("Gender", "unknown")
    neigh_str = data.get("Neighborhood", "unknown")
    subst_str = data.get("Substance", "none")

    # Convert the range to midpoint
    age_val = parse_age(age_str)

    df_input_encoded = build_input_dataframe(age_val, gender_str, neigh_str, subst_str)

    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_proba = model.predict_proba(df_input_encoded)[0][1]
    overdose_probability = float(y_pred_proba)

    confidence = assign_confidence(overdose_probability)
    explanation_str = generate_explanation(overdose_probability, age_val, gender_str, neigh_str, subst_str)

    return jsonify({
        "overdose_probability": overdose_probability,
        "overdose_class": int(y_pred_class),
        "confidence": confidence,
        "explanation": explanation_str
    })

@app.route("/process_query", methods=["POST"])
def process_query():
    """
    Takes a free-text `query` and tries to extract:
      - Age (e.g. '25 to 29' or '25 years old')
      - Gender ('male' or 'female')
      - Neighborhood (partial match)
      - Substance (partial match)

    Then runs a prediction using the model.
    """
    body = request.get_json() or {}
    query = body.get("query", "").lower()

    if not query:
        return jsonify({"error": "Missing query text"}), 400

    # -------------------------
    # A. Extract Numeric Age from "XX years old"
    # -------------------------
    match_age = re.search(r"(\d{1,3})\s?(?:years? old|yrs?)", query)
    numeric_age = None
    if match_age:
        numeric_age_str = match_age.group(1)
        numeric_age = int(numeric_age_str)

    # -------------------------
    # B. Determine age range from numeric age or exact substring
    # -------------------------
    detected_age_range = None
    # 1) If numeric age found, try to map to "X to Y"
    if numeric_age is not None:
        mapped = map_numeric_age_to_range(numeric_age)
        if mapped:
            detected_age_range = mapped

    # 2) Otherwise, see if there's an exact "X to Y" substring in the query
    if not detected_age_range:
        for ar in valid_age_ranges:
            if ar.lower() in query:  # e.g. "30 to 34"
                detected_age_range = ar
                break

    # -------------------------
    # C. Extract Gender (male/female)
    # -------------------------
    # Very naive approach
    gender = None
    if "male" in query:
        gender = "male"
    elif "female" in query:
        gender = "female"

    # -------------------------
    # D. Partial match for Neighborhood & Substance
    # -------------------------
    detected_neighborhood = partial_match(query, valid_neighborhoods)
    detected_substance = partial_match(query, valid_substances)

    # -------------------------
    # E. Check if we got them all
    # -------------------------
    if not detected_age_range or not gender or not detected_neighborhood or not detected_substance:
        return jsonify({"error": "Missing or invalid data (Age, Gender, Neighborhood, Substance)"}), 400

    # Convert age range to midpoint integer
    age_val = parse_age(detected_age_range)

    # -------------------------
    # F. Model prediction
    # -------------------------
    df_input_encoded = build_input_dataframe(age_val, gender, detected_neighborhood, detected_substance)
    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_proba = model.predict_proba(df_input_encoded)[0][1]
    overdose_probability = float(y_pred_proba)

    confidence = assign_confidence(overdose_probability)
    explanation_str = generate_explanation(overdose_probability, age_val, gender, detected_neighborhood, detected_substance)

    return jsonify({
        "overdose_probability": overdose_probability,
        "overdose_class": int(y_pred_class),
        "confidence": confidence,
        "explanation": explanation_str,
        "parsed_data": {
            "AgeRange": detected_age_range,
            "Gender": gender,
            "Neighborhood": detected_neighborhood,
            "Substance": detected_substance
        }
    })

if __name__ == "__main__":
    # Run on 0.0.0.0:6000 so it's externally accessible if your security group/firewall allows it.
    app.run(host="0.0.0.0", port=6000, debug=True)
