import json
import os
import re
import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)  # Use __name__
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------------------------
# 1. Load the trained model (retained for endpoint compatibility)
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
# 3. The main prediction logic (we fake the prediction values)
# --------------------------------------------------------------------
def make_prediction(age_str, gender_str, neigh_str, subst_str):
    # Fake values for demonstration
    overdose_probability = 0.21  # Random fake value
    overdose_class = 0           # Fake value (0 indicates lower risk)
    confidence = "Low"           # Fake confidence
    explanation = (
        "Substance suggests a high overdose risk. Younger age group is correlated with certain substance use patterns (example)."
    )
    return {
        "overdose_probability": overdose_probability,
        "overdose_class": overdose_class,
        "confidence": confidence,
        "explanation": explanation
    }

# --------------------------------------------------------------------
# 4. Helper function to format output with the Gemini LLM (fake call)
# --------------------------------------------------------------------
def format_output_with_gemini(prediction_json):
    """
    Fakes a call to the Gemini API by generating a human-friendly summary based on the prediction JSON.
    The prompt instructs the LLM to analyze the provided input and generated prediction,
    discussing the probability of overdose, class, confidence, and high-risk factors.
    If Winnipeg is mentioned in the input, additional context should be provided.
    """
    # Embedded Gemini API key (insecure for production)
    gemini_api_key = "AIzaSyD9DdnncGKuhhjIMBFXiU7ZVspWtv1qvgI"
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    
    prompt = f"""
You are a knowledgeable and articulate data analyst specializing in substance use risk prediction. Based on the JSON output below, please analyze the input data and generate a detailed, human-friendly summary. Your summary should include:

1. A clear explanation of the predicted overdose probability and what it implies, based on the provided random value.
2. An interpretation of the overdose class (0 or 1) and its significance.
3. A discussion of the confidence level (Low, Medium, or High), and what that means in context.
4. A plain language explanation of any high-risk factors identified in the input.
5. If the input mentions a location like Winnipeg, provide additional local contextual insights.
6. Overall, study the input data carefully and weave a narrative that explains the generated prediction and offers contextual understanding of the overdose risk.
7. Mention the location, name and age -- in ur response again

Below is the JSON output from the substance use prediction model:

{prediction_json}

Return your answer in clear, plain text with organized headings.
"""
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(gemini_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        try:
            data = response.json()
            candidate = data["candidates"][0]
            formatted_text = candidate.get("output") or candidate.get("content")
            if not formatted_text:
                formatted_text = "No formatted output found in Gemini response."
        except Exception as e:
            formatted_text = f"Error parsing Gemini response: {str(e)}"
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
      "Age": "15 to 19",
      "Gender": "Male",
      "Neighborhood": "exchange",
      "Substance": "fentanyl"
    }
    Uses fake prediction logic and pipes the output through Gemini for formatting.
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
    then calls the fake prediction logic and pipes the result through Gemini for formatting.
    """
    body = request.get_json() or {}
    user_text = body.get("text", "").lower()

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Naively extract Age from "NN years old"
    match = re.search(r"(\d{1,3})\s?years?\s?old", user_text)
    if match:
        numeric_age = match.group(1)
        age_str_detected = f"{numeric_age} to {numeric_age}"
    else:
        age_str_detected = "30 to 34"  # fallback

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
