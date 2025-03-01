import json
import os
import re
import requests
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------------------------
# 1. Load the trained model (this remains, though it won't be used for fake predictions)
# --------------------------------------------------------------------
model = joblib.load("model_expanded.pkl")
all_training_cols = getattr(model, "feature_names_in_", None)

# --------------------------------------------------------------------
# 2. Helper function to convert an age range string to a midpoint integer
# --------------------------------------------------------------------
def parse_age(age_str):
    """
    Converts an age range string like '30 to 34' into its midpoint (32).
    Fallback is 30 if parsing fails.
    """
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except Exception:
        return 30  # fallback

# --------------------------------------------------------------------
# 3. The main prediction logic (kept for endpoint compatibility)
# --------------------------------------------------------------------
def make_prediction(age_str, gender_str, neigh_str, subst_str):
    # This function is retained but its output will be ignored for fake predictions.
    # We'll use fake prediction values instead.
    # However, we simulate a prediction structure.
    overdose_probability = 0.21  # Fake value
    overdose_class = 0           # Fake value
    confidence = "Low"           # Fake value
    explanation = (
        "Substance suggests a high overdose risk. Younger age group is correlated with "
        "certain substance use patterns (example)."
    )
    return {
        "overdose_probability": overdose_probability,
        "overdose_class": overdose_class,
        "confidence": confidence,
        "explanation": explanation
    }

# --------------------------------------------------------------------
# 4. Fake Gemini API call: Generate a human-friendly summary without making an actual API call.
# --------------------------------------------------------------------
def format_output_with_gemini(prediction_json):
    """
    Fakes a call to the Gemini API by generating a human-friendly summary based on the prediction JSON.
    """
    try:
        data = json.loads(prediction_json)
        parsed_data = data.get("parsed_data", {})
        prediction = data.get("prediction", {})
        overdose_probability = prediction.get("overdose_probability", 0.0)
        overdose_class = prediction.get("overdose_class", 0)
        confidence = prediction.get("confidence", "Low")
        explanation = prediction.get("explanation", "")
        # Create a nicely formatted summary using the fake values
        fake_summary = (
            f"--- Prediction Summary ---\n"
            f"Overdose Probability: {overdose_probability:.2f}\n"
            f"Overdose Class: {overdose_class} (0 indicates lower risk, 1 indicates higher risk)\n"
            f"Confidence: {confidence}\n\n"
            f"Explanation:\n{explanation}\n\n"
            f"Additional Context: Based on the input data, the risk appears to be moderate. "
            f"Local factors (for example, conditions in Winnipeg or Exchange) may further affect this outcome. "
            f"Please consult a professional for a comprehensive assessment."
        )
        return fake_summary
    except Exception as e:
        return f"Error generating fake summary: {str(e)}"

# --------------------------------------------------------------------
# 5. Flask endpoints
# --------------------------------------------------------------------
@app.route("/")
def home():
    return "Fake Substance Use Prediction API using Gemini LLM is running!"

@app.route("/predict_expanded", methods=["POST"])
def predict_expanded():
    """
    Expects JSON with:
    {
      "Age": "15 to 19",
      "Gender": "Male",
      "Neighborhood": "exchange",
      "Substance": "fentanyl"
    }
    Uses the fake prediction logic and Gemini faking to generate output.
    """
    data = request.get_json() or {}
    age_str = data.get("Age")
    gender_str = data.get("Gender")
    neigh_str = data.get("Neighborhood")
    subst_str = data.get("Substance")
    
    if not all([age_str, gender_str, neigh_str, subst_str]):
        return jsonify({"error": "Missing required fields: Age, Gender, Neighborhood, Substance"}), 400

    result = make_prediction(age_str, gender_str, neigh_str, subst_str)
    return jsonify(result)

@app.route("/predict_from_text", methods=["POST"])
def predict_from_text():
    """
    Expects JSON like:
    {
      "text": "What is the substance use of a 15 to 19 years old male in Exchange?"
    }
    Parses the free-text (naively) to extract Age, Gender, Neighborhood, and Substance,
    then uses the fake prediction logic and returns a Gemini-faked formatted summary.
    """
    body = request.get_json() or {}
    user_text = body.get("text", "").lower()

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Naively extract Age from text using regex
    match = re.search(r"(\d{1,3})\s?years?\s?old", user_text)
    if match:
        numeric_age = match.group(1)
        age_str_detected = f"{numeric_age} to {numeric_age}"
    else:
        age_str_detected = "30 to 34"  # fallback if not found

    # Naively extract Gender
    gender_detected = "male" if "male" in user_text else ("female" if "female" in user_text else "unknown")

    # Naively extract Neighborhood from a predefined list
    neighborhood_detected = "unknown"
    possible_neighborhoods = ["robertson", "downtown", "brooklyn", "queens", "winnipeg", "exchange"]
    for neigh in possible_neighborhoods:
        if neigh in user_text:
            neighborhood_detected = neigh
            break

    # Naively extract Substance from a predefined list
    substance_detected = "none"
    possible_substances = ["fentanyl", "opioid", "heroin", "alcohol", "cocaine", "meth"]
    for s in possible_substances:
        if s in user_text:
            substance_detected = s
            break

    # Call the fake prediction logic
    result = make_prediction(
        age_str_detected,
        gender_detected,
        neighborhood_detected,
        substance_detected
    )

    parsed_data = {
        "Age": age_str_detected,
        "Gender": gender_detected,
        "Neighborhood": neighborhood_detected,
        "Substance": substance_detected
    }
    
    # Build a JSON string from the parsed data and prediction to send to our fake Gemini function
    prediction_json = json.dumps({
        "parsed_data": parsed_data,
        "prediction": result
    }, indent=2)
    
    formatted_output = format_output_with_gemini(prediction_json)
    
    return jsonify({
        "parsed_data": parsed_data,
        "prediction": result,
        "formatted_output": formatted_output
    })

# --------------------------------------------------------------------
# Run the Flask server on port 8080
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
