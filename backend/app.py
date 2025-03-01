import joblib
import pandas as pd
import re

from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------
# 1. Setup Flask
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# -------------------------
# 2. Load Model and Dataset
# -------------------------
model = joblib.load("model_expanded.pkl")
all_training_cols = getattr(model, "feature_names_in_", None)

# Load dataset.csv to memory
df_dataset = pd.read_csv("Substance_Use_20250301.csv")  # Change to your actual path or method
# Convert columns to lower-case or standard forms where needed
df_dataset["Neighbourhood"] = df_dataset["Neighbourhood"].str.lower()
df_dataset["Substance"] = df_dataset["Substance"].str.lower()

# Extract possible unique sets to validate user inputs
valid_neighborhoods = set(df_dataset["Neighbourhood"].dropna().unique())
valid_substances = set(df_dataset["Substance"].dropna().unique())
valid_age_ranges = set(df_dataset["Age"].dropna().unique())  # e.g., '35 to 39'

# -------------------------
# 3. Helper Functions
# -------------------------
def parse_age(age_str):
    """Parses the 'Age' string like '30 to 34' and returns an integer midpoint."""
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return 30  # fallback

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

    # One-hot encode needed columns
    df_input_encoded = pd.get_dummies(df_input, columns=["Neighborhood", "Substance"],
                                      prefix=["neigh", "subst"])
    # Align with training columns
    if all_training_cols is not None:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[all_training_cols]

    return df_input_encoded

def generate_explanation(overdose_probability, age_val, gender_str, neigh_str, subst_str):
    """Generate a textual explanation based on input features and outcome."""
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
    """Assign confidence labels based on the probability."""
    if prob > 0.7:
        return "High"
    elif prob > 0.3:
        return "Medium"
    else:
        return "Low"

def is_valid_value(value, valid_set):
    """Check if value is in the known set (case-insensitive)."""
    return value.lower() in (x.lower() for x in valid_set)

# -------------------------
# 4. Endpoints
# -------------------------
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

    # Extract fields from JSON
    age_str = data.get("Age", "30 to 34")
    gender_str = data.get("Gender", "unknown")
    neigh_str = data.get("Neighborhood", "unknown")
    subst_str = data.get("Substance", "none")

    # Convert age from range to midpoint integer
    age_val = parse_age(age_str)

    # Build input DataFrame, align columns
    df_input_encoded = build_input_dataframe(age_val, gender_str, neigh_str, subst_str)

    # Predict
    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_proba = model.predict_proba(df_input_encoded)[0][1]
    overdose_probability = float(y_pred_proba)

    # Confidence & Explanation
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
    Endpoint that takes a free-text `query` and tries to extract:
    - Age (e.g. '25 years old' or an exact match '30 to 34'),
    - Gender ('male' or 'female'),
    - Neighborhood,
    - Substance

    Then it forwards the parsed data to the model.
    """
    body = request.get_json() or {}
    query = body.get("query", "").lower()  # the free-text query

    if not query:
        return jsonify({"error": "Missing query text"}), 400

    # -------------------------
    # 4a. Extract Age
    # -------------------------
    # Option A: if your dataset only has range strings like "30 to 34" or "35 to 39",
    # you might do a simple check for these substrings in the query.
    # Or do a regex approach for "(\d{1,3})\s?(?:years? old|yrs?)" etc.
    #
    # Here, we'll do a naive approach: check each valid_age_range in the query
    # and pick the first that appears. If none found, age = None
    # (You can refine as needed.)
    detected_age_range = None
    for ar in valid_age_ranges:
        # e.g. "30 to 34"
        if ar.lower() in query:
            detected_age_range = ar
            break

    # Alternatively, if you want to detect "25 years old" patterns, do:
    match_age = re.search(r"(\d{1,3})\s?(?:years? old|yrs?)", query)
    if match_age:
        # see if the user wrote "25 years old" but your dataset expects something like "25 to 29"
        # you'd have to map numeric ages to your dataset's nearest range
        # For now, let's just override `detected_age_range` with the numeric
        numeric_age_str = match_age.group(1)
        # We won't do a strict "is this in the dataset" check, since dataset has ranges
        # We'll do a fallback. Or you can skip this if your dataset strictly has "25 to 29"
        detected_age_range = f"{numeric_age_str} to {numeric_age_str}"

    # -------------------------
    # 4b. Extract Gender
    # -------------------------
    # If you only handle male/female:
    gender = None
    if "male" in query:
        gender = "male"
    elif "female" in query:
        gender = "female"

    # -------------------------
    # 4c. Extract Neighborhood
    # -------------------------
    # We'll look for any valid neighborhood in the query:
    # (In a real system, you might do fuzzy matching or an NLP approach.)
    detected_neighborhood = None
    for neigh in valid_neighborhoods:
        if neigh in query:
            detected_neighborhood = neigh
            break

    # -------------------------
    # 4d. Extract Substance
    # -------------------------
    # We'll look for any valid substance in the query:
    detected_substance = None
    for sub in valid_substances:
        if sub in query:
            detected_substance = sub
            break

    # -------------------------
    # 4e. Validate or Error
    # -------------------------
    if not detected_age_range or not gender or not detected_neighborhood or not detected_substance:
        return jsonify({"error": "Missing or invalid data (Age, Gender, Neighborhood, Substance)"}), 400

    # Everything is extracted and valid. We now have age, gender, neighborhood, substance.
    # Let's pass them along to the same code that /predict_expanded uses.
    # We can either:
    #    A) Call model right here
    #    B) Or forward a request to /predict_expanded
    #
    # For simplicity, let's call the model right here.

    age_val = parse_age(detected_age_range)
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
            "Age": detected_age_range,
            "Gender": gender,
            "Neighborhood": detected_neighborhood,
            "Substance": detected_substance
        }
    })

# -------------------------
# 5. Run the Flask App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
