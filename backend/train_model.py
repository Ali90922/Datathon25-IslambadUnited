# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def parse_age(age_str):
    """
    Converts an age range like '30 to 34' into a numeric estimate (e.g. 32).
    If parsing fails, returns None.
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

    # 2. Minimal data cleaning & feature engineering

    # Parse Age to a numeric midpoint
    df["AgeNumeric"] = df["Age"].apply(parse_age)

    # Encode Gender as numeric (male=1, otherwise=0)
    # Adjust logic if you have more gender categories
    df["GenderNum"] = df["Gender"].apply(
        lambda g: 1 if str(g).strip().lower() == "male" else 0
    )

    # Drop rows missing these new columns
    df = df.dropna(subset=["AgeNumeric", "GenderNum"])

    # 3. Define a target variable
    # For demo: let's classify whether "Substance" is "Opioids" or not ("IsOpioid")
    df["IsOpioid"] = df["Substance"].apply(
        lambda s: 1 if str(s).strip().lower() == "opioids" else 0
    )

    # Features & target
    X = df[["AgeNumeric", "GenderNum"]]
    y = df["IsOpioid"]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train a Logistic Regression (good for classification)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 6. Evaluate quickly
    score = model.score(X_test, y_test)
    print(f"LogisticRegression accuracy on test set: {score:.4f}")

    # 7. Save the model
    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_and_save_model()

