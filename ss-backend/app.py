import json
import re
import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)  # Use __name__
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------------------------
# 1. Load the trained model (retained for endpoint compatibility)
# --------------------------------------------------------------------
model = joblib.load("model_calibrated.pkl")
all_training_cols = getattr(model, "feature_names_in_", None)
all_neighborhoods = ["Agassiz", "Airport", "Alpine Place", "Amber Trails", "Archwood", "Armstrong Point", "Assiniboia Downs", "Assiniboine Park", "Beaumont", "Betsworth", "Birchwood", "Booth", "Bridgwater Centre", "Bridgwater Forest", "Bridgwater Lakes", "Bridgwater Trails", "Broadway-Assiniboine", "Brockville", "Brooklands", "Bruce Park", "Buchanan", "Buffalo", "Burrows Central", "Burrows-Keewatin", "Canterbury Park", "Centennial", "Central Park", "Central River Heights", "Central St. Boniface", "Chalmers", "Chevrier", "China Town", "Civic Centre", "Cloutier Drive", "Colony", "Crescent Park", "Crescentwood", "Crestview", "Dakota Crossing", "Daniel McIntyre", "Daniel Mcintyre", "Deer Lodge", "Dufferin", "Dufferin Industrial", "Dufresne", "Dugald", "Eaglemere", "Earl Grey", "East Elmwood", "Ebby-Wentworth", "Edgeland", "Elm Park", "Elmhurst", "Eric Coy", "Exchange District", "Fairfield Park", "Fort Richmond", "Fraipont", "Garden City", "Glendale", "Glenelm", "Glenwood", "Grant Park", "Grassie", "Heritage Park", "Holden", "Inkster Gardens", "Inkster Industrial Park", "Inkster-Faraday", "Island Lakes", "J. B. Mitchell", "Jameswood", "Jefferson", "Kensington", "Kern Park", "Kil-Cona Park", "Kildare-Redonda", "Kildonan Crossing", "Kildonan Drive", "Kildonan Park", "King Edward", "Kingston Crescent", "Kirkfield", "La Barriere", "Lavalee", "Legislature", "Leila North", "Leila-McPhillips Triangle", "Leila-Mcphillips Triangle", "Linden Ridge", "Linden Woods", "Logan-C.P.R.", "Lord Roberts", "Lord Selkirk Park", "Luxton", "Maginot", "Mandalay West", "Maple Grove Park", "Margaret Park", "Marlton", "Mathers", "Maybank", "McLeod Industrial", "McMillan", "Mcleod Industrial", "Mcmillan", "Meadowood", "Meadows", "Melrose", "Minnetonka", "Minto", "Mission Gardens", "Mission Industrial", "Montcalm", "Munroe East", "Munroe West", "Murray Industrial Park", "Mynarski", "Niakwa Park", "Niakwa Place", "Norberry", "Normand Park", "North Inkster Industrial", "North Point Douglas", "North River Heights", "North St. Boniface", "North Transcona Yards", "Norwood East", "Norwood West", "Oak Point Highway", "Old Tuxedo", "Omand's Creek Industrial", "Pacific Industrial", "Parc La Salle", "Parker", "Peguis", "Pembina Strip", "Perrault", "Point Road", "Polo Park", "Portage & Main", "Portage-Ellice", "Prairie Pointe", "Pulberry", "Radisson", "Regent", "Richmond Lakes", "Richmond West", "Ridgedale", "Ridgewood South", "River East", "River Park South", "River West Park", "River-Osborne", "Riverbend", "Rivergrove", "Riverview", "Robertson", "Roblin Park", "Rockwood", "Roslyn", "Rosser-Old Kildonan", "Rossmere-A", "Rossmere-B", "Royalwood", "Sage Creek", "Sargent Park", "Saskatchewan North", "Seven Oaks", "Shaughnessy Park", "Silver Heights", "Sir John Franklin", "South Point Douglas", "South Pointe", "South Portage", "South River Heights", "South Tuxedo", "Southboine", "Southdale", "Southland Park", "Spence", "Springfield North", "Springfield South", "St. Boniface Industrial Park", "St. George", "St. James Industrial", "St. John's", "St. John's Park", "St. Matthews", "St. Norbert", "St. Vital Centre", "St. Vital Perimeter South", "Stock Yards", "Sturgeon Creek", "Symington Yards", "Talbot-Grey", "Templeton-Sinclair", "The Forks", "The Maples", "Tissot", "Transcona North", "Transcona South", "Transcona Yards", "Trappistes", "Turnbull Drive", "Tuxedo", "Tuxedo Industrial", "Tyndall Park", "Tyne-Tees", "University", "Valhalla", "Valley Gardens", "Varennes", "Varsity View", "Vialoux", "Victoria Crescent", "Victoria West", "Vista", "Waverley Heights", "Wellington Crescent", "West Alexander", "West Broadway", "West Fort Garry Industrial", "West Kildonan Industrial", "West Wolseley", "Westdale", "Weston", "Weston Shops", "Westwood", "Whyte Ridge", "Wildwood", "Wilkes South", "William Whyte", "Windsor Park", "Wolseley", "Woodhaven", "Worthington"]
all_substances = ["Alcohol", "Cocaine", "Crystal Meth", "Marijuana", "Opioids"]
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
    # Load model and feature columns
    model = joblib.load("model_calibrated.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    
    # Initialize input DataFrame with all zeroes
    input_df = pd.DataFrame(columns=feature_cols)
    input_df.loc[0] = 0
    
    # Process input values
    age_numeric = parse_age(age_str)
    input_df.at[0, "AgeNumeric"] = age_numeric
    gender_num = 1 if gender_str.lower() == "female" else 0
    input_df.at[0, "GenderNum"] = gender_num
    
    # One-hot encode neighborhood
    neigh_col = f"neigh_{neigh_str.lower()}"
    if neigh_col in input_df.columns:
        input_df.at[0, neigh_col] = 1
    
    # One-hot encode substance
    subst_col = f"subst_{subst_str.lower()}"
    if subst_col in input_df.columns:
        input_df.at[0, subst_col] = 1
    
    # Adjusted risk class based on substance
    substance = subst_str.lower()
    if substance in ["opioids", "crystal meth", "cocaine"]:
        overdose_class = 2  # High risk
    elif substance == "alcohol":
        overdose_class = 1   # Medium risk
    else:
        overdose_class = 0   # Low risk
    
    # Use the model to predict probability
    probs = model.predict_proba(input_df)[0]
    
    # Set probability based on model prediction, rather than hardcoded
    overdose_probability = probs[overdose_class]  # Use the actual model's predicted probability
    
    # Determine confidence based on overdose probability
    if overdose_probability > 0.7:
        confidence = "High"
    elif overdose_probability > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Now return the results with proper class and probability
    return {
        "overdose_probability": round(overdose_probability, 2),
        "overdose_class": overdose_class,
        "confidence": confidence
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
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    
    prompt = f"""
    You are a knowledgeable and articulate data analyst specializing in substance use risk prediction. Based on the JSON output below, please analyze the input data and generate a detailed, human-friendly summary. Your summary should include:

2. An overdose probability, the percentage in bold, very clearly formatted.
1. An interpretation of the overdose class (0, 3, or 10) and its significance in simple English, where:
   - 0 represents low risk
   - 1 represents medium risk (typically associated with alcohol)
   - 2 represents high risk (typically associated with opioids, crystal meth, or cocaine)
2. A discussion of the confidence level (Low, Medium, or High), and what that means in context.
3. A plain language explanation of any high-risk factors identified in the input.
4. If the input mentions a location like Winnipeg, provide additional local contextual insights.
5. Mention the age, location, gender, and substance (all starting with a capital letter e.g. "Male") -- in ur response again CLEARLY at the start of your message in a list format. If the age range mentioned is just 1 age, e.g. 13 to 13, just use 13 instead of 13 to 13. DO NOT include Winnipeg in the location, just the name of the area.
6. Use bold headings, separate concerns, and make your output readable.

You are presenting this information to a user in markdown format, please make sure it is readable, not too long, concise, and something that makes sense. Do NOT use sentences like "Okay, here's an analysis of the substance use prediction model's output, presented in a user-friendly markdown format:" to start or end your text, the user does not need to know how you are getting this output. Also, always end your message with a disclaimer: "This analysis is based solely on the provided data and should not be considered a definitive diagnosis or prediction. Professional medical and psychological evaluation is essential for a comprehensive assessment and personalized recommendations." in italics under a --- line in markdown.

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
    print(data);
    
    data = request.get_json() or {}
    age_str = data.get("Age")
    gender_str = data.get("Gender")
    neigh_str = data.get("Neighborhood")
    subst_str = data.get("Substance")
    
    if not all([age_str, gender_str, neigh_str, subst_str]):
        return jsonify({"error": "Missing required fields: Age, Gender, Neighborhood, Substance"}), 400

    result = make_prediction(age_str, gender_str, neigh_str, subst_str)
    return jsonify(result)

def extract_age(user_text):
    # Regex to match a single age, e.g., "19 years old"
    match_single = re.search(r"(\d{1,3})\s?years?\s?old", user_text)
    
    if match_single:
        # Return the age as "X to X", e.g., "19 to 19"
        return f"{match_single.group(1)} to {match_single.group(1)}"
    
    # Fallback if no age is found
    return "15 to 19"  # Default fallback age range

def extract_gender(user_text):
    if "female" in user_text.lower():
        return "female"
    elif "male" in user_text.lower():
        return "male"
    return "unknown"

def extract_neighborhood(user_text):
    for neigh in all_neighborhoods:
        if neigh.lower() in user_text.lower():
            return neigh
    return "unknown"

def extract_substance(user_text):
    for s in all_substances:
        if s.lower() in user_text.lower():
            return s
    return "none"

@app.route("/predict_from_text", methods=["POST"])
def predict_from_text():
    body = request.get_json() or {}
    user_text = body.get("text", "").lower()

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    age_str_detected = extract_age(user_text)
    gender_detected = extract_gender(user_text)
    neighborhood_detected = extract_neighborhood(user_text)
    substance_detected = extract_substance(user_text)

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
# Run the Flask server on port 8082
# --------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
