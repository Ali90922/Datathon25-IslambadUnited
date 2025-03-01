import joblib
import pandas as pd
import re
import json
import os
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------------------------
# 1. Load the trained model
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
# 3. The main prediction logic
# --------------------------------------------------------------------
def make_prediction(age_str, gender_str, neigh_str, subst_str):
    # Convert the age range to a numeric midpoint
    age_val = parse_age(age_str)
    # Encode gender: "male" is 1, anything else is 0
    gender_num = 1 if gender_str.strip().lower() == "male" else 0

    # Build the input DataFrame
    df_input = pd.DataFrame({
        "AgeNumeric": [age_val],
        "GenderNum": [gender_num],
        "Neighborhood": [neigh_str.strip().lower()],
        "Substance": [subst_str.strip().lower()]
    })

    # One-hot encode categorical features
    df_input_encoded = pd.get_dummies(
        df_input,
        columns=["Neighborhood", "Substance"],
        prefix=["neigh", "subst"]
    )

    # Ensure the DataFrame has the same columns as the model training
    if all_training_cols is not None:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[all_training_cols]

    # Get predictions
    y_pred_class = model.predict(df_input_encoded)[0]
    y_pred_proba = model.predict_proba(df_input_encoded)[0][1]
    overdose_probability = float(y_pred_proba)

    # Determine confidence level
    if overdose_probability > 0.7:
        confidence = "High"
    elif overdose_probability > 0.3:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Generate a simple explanation based on input features
    explanation = []
    if "opioid" in subst_str.lower() or "fentanyl" in subst_str.lower():
        explanation.append("Substance suggests a high overdose risk.")
    if neigh_str.lower() == "downtown":
        explanation.append("Downtown area might have higher incidents in this dataset.")
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
# 4. Helper function to format output with the Gemini LLM
# --------------------------------------------------------------------
def format_output_with_gemini(prediction_json):
    """
    Sends the prediction JSON to the Gemini API and returns a nicely formatted summary.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return "Gemini API key not set."
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    
    prompt = f"""
You are an expert data summarizer. Given the JSON output from a substance use prediction model:

{prediction_json}

Please provide a nicely formatted, human-friendly summary that explains:
- The predicted overdose probability.
- The overdose class and its meaning.
- The confidence level and its implications.
- A plain language explanation of any high-risk factors.
Include any contextual information that may help understand the prediction.
Return your answer in plain text.
"""
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(gemini_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        try:
            # Assuming the response returns a structure with a "candidates" list and "output" key
            formatted_text = data["candidates"][0]["output"]
        except Exception as e:
            formatted_text = "Error parsing Gemini response."
        return formatted_text
    else:
        return f"Error from Gemini: {response.status_code} {response.text}"

# --------------------------------------------------------------------
# 5. Flask endpoints
# --------------------------------------------------------------------
@app.route("/")
def home():
    return "Enhanced Substance Use Prediction API is running!"

@app.route("/predict_expanded", methods=["POST"])
def predict_expanded():
    """
    Expects JSON with:
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
    result = make_prediction(age_str, gender_str, neigh_str, subst_str)
    return jsonify(result)

@app.route("/predict_from_text", methods=["POST"])
def predict_from_text():
    """
    Expects JSON like:
    {
      "text": "What is the substance use of a 35 years old female in Robertson?"
    }
    Parses the free-text to extract Age, Gender, Neighborhood, and Substance,
    then calls the prediction function and pipes the result through Gemini for formatting.
    """
    body = request.get_json() or {}
    user_text = body.get("text", "").lower()

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Parse Age from "NN years old"
    match = re.search(r"(\d{1,3})\s?years?\s?old", user_text)
    if match:
        numeric_age = match.group(1)
        age_str_detected = f"{numeric_age} to {numeric_age}"
    else:
        age_str_detected = "30 to 34"

    # Parse Gender
    gender_detected = "unknown"
    if "male" in user_text:
        gender_detected = "male"
    elif "female" in user_text:
        gender_detected = "female"

    # Parse Neighborhood (naive approach)
    neighborhood_detected = "unknown"
    possible_neighborhoods = ["robertson", "downtown", "brooklyn", "queens"]
    for neigh in possible_neighborhoods:
        if neigh in user_text:
            neighborhood_detected = neigh
            break

    # Parse Substance (naive approach)
    substance_detected = "none"
    possible_substances = ["fentanyl", "opioid", "heroin", "alcohol", "cocaine", "meth"]
    for s in possible_substances:
        if s in user_text:
            substance_detected = s
            break

    # Call the prediction function with parsed fields
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
    
    # Build a JSON string from the parsed data and prediction
    prediction_json = json.dumps({
        "parsed_data": parsed_data,
        "prediction": result
    }, indent=2)
    
    # Get the formatted output from Gemini
    formatted_output = format_output_with_gemini(prediction_json)
    
    return jsonify({
        "parsed_data": parsed_data,
        "prediction": result,
        "formatted_output": formatted_output
    })

# --------------------------------------------------------------------
# 6. Run the Flask server on a safe port (e.g., 8080)
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
