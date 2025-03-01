# train_model_expanded.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def parse_age(age_str):
    """
    Convert an age range like '30 to 34' into a numeric midpoint (e.g., 32).
    If parsing fails, return None.
    """
    try:
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return None

def train_and_save_model():
    # 1. Load the dataset
    df = pd.read_csv("Substance_Use_20250301.csv")
    
    # ------------------------------------------------
    # 2. Minimal data cleaning & feature engineering
    # ------------------------------------------------
    
    # --- Age ---
    # Replace missing or malformed Age with a default "30 to 34", then parse
    df["Age"] = df["Age"].fillna("30 to 34")  # fallback if Age is missing
    df["Age"] = df["Age"].astype(str).str.strip()  # ensure string
    df["AgeNumeric"] = df["Age"].apply(parse_age)

    # --- Gender ---
    # Default to "unknown" if missing
    df["Gender"] = df["Gender"].fillna("unknown").astype(str).str.lower().str.strip()
    # We'll interpret "male"=1, everything else=0 (including female, unknown, etc.)
    df["GenderNum"] = df["Gender"].apply(lambda g: 1 if g == "male" else 0)

    # --- Neighborhood ---
    # If your CSV uses "Ward" or "Neighbourhood" as a column, rename:
    if "Neighborhood" not in df.columns:
        # Example if your dataset has "Ward" instead:
        if "Ward" in df.columns:
            df.rename(columns={"Ward": "Neighborhood"}, inplace=True)
        else:
            # If truly missing, create a placeholder column
            df["Neighborhood"] = "unknown"
    df["Neighborhood"] = df["Neighborhood"].fillna("unknown").astype(str).str.lower().str.strip()

    # --- Substance ---
    if "Substance" not in df.columns:
        # If your dataset uses "DrugType" or something else, rename accordingly
        if "DrugType" in df.columns:
            df.rename(columns={"DrugType": "Substance"}, inplace=True)
        else:
            df["Substance"] = "none"
    df["Substance"] = df["Substance"].fillna("none").astype(str).str.lower().str.strip()

    # ------------------------------------------------
    # 3. Define your outcome / target variable
    # ------------------------------------------------
    # For demonstration, let's assume we want to predict whether it's an "overdose" or not.
    # We'll create a binary label "IsOverdose" = 1 if "opioid" is mentioned in Substance, else 0.
    # Adjust as needed for your actual outcome.
    df["IsOverdose"] = df["Substance"].apply(lambda s: 1 if "opioid" in s else 0)

    # ------------------------------------------------
    # 4. Drop rows missing essential numeric features
    # ------------------------------------------------
    # If AgeNumeric is None, we can't proceed. 
    df = df.dropna(subset=["AgeNumeric"])

    # ------------------------------------------------
    # 5. One-Hot Encode Neighborhood & Substance
    # ------------------------------------------------
    # We'll add prefix "neigh_" or "subst_" to avoid column collisions
    df_encoded = pd.get_dummies(
        df,
        columns=["Neighborhood", "Substance"],
        prefix=["neigh", "subst"],
        drop_first=False  # keep all columns, or set True to reduce dimensionality
    )

    # ------------------------------------------------
    # 6. Create feature matrix X and target y
    # ------------------------------------------------
    # Start with basic numeric columns
    feature_cols = ["AgeNumeric", "GenderNum"]
    # Then add the one-hot columns that start with "neigh_" or "subst_"
    feature_cols += [c for c in df_encoded.columns if c.startswith("neigh_") or c.startswith("subst_")]

    X = df_encoded[feature_cols]
    y = df_encoded["IsOverdose"]

    # ------------------------------------------------
    # 7. Train-Test Split
    # ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------------------------------------
    # 8. Train a RandomForestClassifier
    # ------------------------------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"RandomForest accuracy on test set: {accuracy:.4f}")

    # ------------------------------------------------
    # 9. Save the model
    # ------------------------------------------------
    joblib.dump(model, "model_expanded.pkl")
    print("Model saved to model_expanded.pkl")

if __name__ == "__main__":
    train_and_save_model()

