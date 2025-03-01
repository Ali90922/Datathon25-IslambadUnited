import json
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Gemini API key and URL (insecurely embedded for this example)
GEMINI_API_KEY = "AIzaSyD9DdnncGKuhhjIMBFXiU7ZVspWtv1qvgI"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def generate_fake_prediction_structured(age, gender, neighborhood, substance):
    prompt = f"""
You are an expert in substance use risk prediction. Using the following input data:
- Age: {age}
- Gender: {gender}
- Neighborhood: {neighborhood}
- Substance: {substance}

Please generate a fake prediction result in JSON format with the following keys:
- "overdose_probability": a random float between 0 and 1 (for example, 0.21)
- "overdose_class": either 0 or 1
- "confidence": one of "Low", "Medium", or "High"
- "explanation": a detailed, human-friendly explanation of the prediction including any relevant contextual insights.

Ensure that your output is valid JSON.
"""
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            data = response.json()
            candidate = data["candidates"][0]
            output_text = candidate.get("output") or candidate.get("content")
            if output_text:
                return json.loads(output_text)
            else:
                return {"error": "No output provided by Gemini."}
        except Exception as e:
            return {"error": f"Error parsing Gemini response: {str(e)}"}
    else:
        return {"error": f"Gemini API error: {response.status_code} {response.text}"}

def generate_fake_prediction_free(text):
    prompt = f"""
You are an expert in substance use risk prediction. Given the following free-text query:
{text}

Please extract any details if available and generate a fake prediction result in JSON format with the following keys:
- "Age": a plausible age range (e.g., "15 to 19")
- "Gender": either "male" or "female"
- "Neighborhood": a neighborhood name (if mentioned)
- "Substance": the substance in question
- "overdose_probability": a random float between 0 and 1,
- "overdose_class": either 0 or 1,
- "confidence": one of "Low", "Medium", or "High",
- "explanation": a detailed, human-friendly explanation of the prediction including relevant context.

Ensure your output is valid JSON.
"""
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            data = response.json()
            candidate = data["candidates"][0]
            output_text = candidate.get("output") or candidate.get("content")
            if output_text:
                return json.loads(output_text)
            else:
                return {"error": "No output provided by Gemini."}
        except Exception as e:
            return {"error": f"Error parsing Gemini response: {str(e)}"}
    else:
        return {"error": f"Gemini API error: {response.status_code} {response.text}"}

# --------------------------------------------------------------------
# Flask Endpoints
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
    Uses Gemini LLM to generate fake prediction values and explanation.
    """
    data = request.get_json() or {}
    age = data.get("Age")
    gender = data.get("Gender")
    neighborhood = data.get("Neighborhood")
    substance = data.get("Substance")
    
    if not all([age, gender, neighborhood, substance]):
        return jsonify({"error": "Missing required fields: Age, Gender, Neighborhood, Substance"}), 400

    result = generate_fake_prediction_structured(age, gender, neighborhood, substance)
    return jsonify(result)

@app.route("/predict_from_text", methods=["POST"])
def predict_from_text():
    """
    Expects JSON with:
    {
      "text": "What is the substance use of a 15 to 19 years old male in Exchange?"
    }
    Uses the free-text query to generate fake prediction values and a human-friendly explanation via Gemini LLM.
    """
    data = request.get_json() or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = generate_fake_prediction_free(text)
    return jsonify(result)

# --------------------------------------------------------------------
# Run the Flask server on port 8080
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
