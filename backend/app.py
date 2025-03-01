import joblib
import pandas as pd
import re
import json
import os
import requests
import spacy

from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize spaCy English model
nlp = spacy.load("en_core_web_sm")

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
    # Encode gender: "male" is 1, others 0
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

    # Ensure the DataFrame has the same columns as used during training
    if all_training_cols is not None:
        for col in all_training_cols:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[all_training_cols]

    # Get predictions from the model
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

    # Generate a simple explanation
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
# 4. Improved free-text parser using spaCy
# --------------------------------------------------------------------
def parse_free_text(user_text):
    """
    Uses spaCy to parse the free-text input and extract:
      - Age (as a number)
      - Gender ("male" or "female")
      - Neighborhood (from a predefined list)
      - Substance (from a predefined list)
    Returns a tuple (age, gender, neighborhood, substance) or None if any cannot be extracted.
    """
    doc = nlp(user_text)
    age = None
    gender = None
    neighborhood = None
    substance = None

    # Extract age: look for numeric tokens followed by "year(s)" or "yr(s)"
    for i, token in enumerate(doc):
        if token.like_num:
            # Check if the next token (if any) is a variant of "year"
            if i + 1 < len(doc) and doc[i + 1].text.lower() in ["year", "years", "yr", "yrs"]:
                age = token.text
                break

    # Extract gender
    if "male" in user_text:
        gender = "male"
    elif "female" in user_text:
        gender = "female"

    # Predefined lists for neighborhood and substance
    possible_neighborhoods = ["robertson", "downtown", "brooklyn", "queens", "winnipeg"]
    for token in doc:
        token_text = token.text.lower()
        if token_text in possible_neighborhoods:
            neighborhood = token_text
            break

    possible_substances = ["fentanyl", "opioid", "heroin", "alcohol", "cocaine", "meth"]
    for token in doc:
        token_text = token.text.lower()
        if token_text in possible_substances:
            substance = token_text
            break

    return age, gender, neighborhood, substance

# --------------------------------------------------------------------
# 5. Helper function to format output with the Gemini LLM
# --------------------------------------------------------------------
def format_output_with_gemini(prediction_json):
    """
    Sends the prediction JSON to the Gemini API and returns a nicely formatted summary.
    """
    # Embedded Gemini API key (insecure for production)
    gemini_api_key = "AIzaSyD9DdnncGKuhhjIMBFXiU7ZVspWtv1qvgI"
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    
    prompt = f"""
You are an expert data summarizer and storyteller. Given the JSON output from a substance use prediction model:

{prediction_json}

Please provide a detailed, human-friendly summary that explains:
- The predicted overdose probability,
- The overdose class and its meaning,
- The confidence level and its implications,
- A plain language explanation of any high-risk factors,
- And additional contextual insights (for example, local context related to Winnipeg if applicable).

Format the summary with clear headings and an engaging tone.
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
            candidate = data["candidates"][0]
            # Try both "output" and "content" keys
            formatted_text = candidate.get("output") or candidate.get("content")
            if not formatted_text:
                formatted_text = "No formatted output found in Gemini response."
        except Exception as e:
            formatted_text = f"Error parsing Gemini response: {str(e)}"
        return formatted_text
    else:
        return f"Error from Gemini: {response.status_code} {response.text}"

# --------------------------------------------------------------------
# 6. Flask endpoints
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
    age_str = data.get("Age")
    gender_str = data.get("Gender")
    neigh_str = data.get("Neighborhood")
    subst_str = data.get("Substance")
    
    # Validate that all fields are provided
    if not all([age_str, gender_str, neigh_str, subst_str]):
        return jsonify({"error": "Missing one or more required fields: Age, Gender, Neighborhood, Substance"}), 400
    
    result = make_prediction(age_str, gender_str, neigh_str, subst_str)
    return jsonify(result)

@app.route("/predict_from_text", methods=["POST"])
def predict_from_text():
    """
    Expects JSON like:
    {
      "text": "What is the substance use of a 35 years old female in Winnipeg?"
    }
    Uses spaCy to parse the free-text input to extract Age, Gender, Neighborhood, and Substance.
    Then calls the prediction function and pipes the result through Gemini for formatting.
    """
    body = request.get_json() or {}
    user_text = body.get("text", "")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Use spaCy to parse the text
    age, gender, neighborhood, substance = parse_free_text(user_text.lower())
    
    # Check that all values were successfully extracted
    if not age:
        return jsonify({"error": "Could not extract age from input."}), 400
    if not gender:
        return jsonify({"error": "Could not extract gender from input."}), 400
    if not neighborhood:
        return jsonify({"error": "Could not extract a valid neighborhood from input."}), 400
    if not substance:
        return jsonify({"error": "Could not extract a valid substance from input."}), 400

    age_str_detected = f"{age} to {age}"  # Use the extracted age as a range

    result = make_prediction(
        age_str_detected,
        gender,
        neighborhood,
        substance
    )

    parsed_data = {
        "Age": age_str_detected,
        "Gender": gender,
        "Neighborhood": neighborhood,
        "Substance": substance
    }
    
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
# 7. Run the Flask server on a safe port (e.g., 8080)
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
