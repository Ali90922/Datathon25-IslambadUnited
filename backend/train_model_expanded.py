import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def classify_overdose_risk(substance):
    substance = substance.lower()  # Convert to lowercase for consistency
    if substance in ["opioids", "crystal meth", "cocaine"]:
        return 2  # High risk
    elif substance == "alcohol":
        return 1  # Medium risk 
    else:
        return 0  # Low risk

def parse_age(age_str):
    """
    Convert an age range like '30 to 34' into a numeric midpoint (e.g., 32).
    If parsing fails, return None.
    """
    try:
        if pd.isna(age_str):
            return None
        parts = age_str.split(" to ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) // 2
    except:
        return None

def train_and_save_model():
    try:
        print("Starting the training process...")

        # 1. Load data
        df = pd.read_csv("Substance_Use_20250301.csv")
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # 2. Clean + Normalize
        df["Age"] = df["Age"].fillna("30 to 34").astype(str).str.strip()
        df["AgeNumeric"] = df["Age"].apply(parse_age)

        df["Gender"] = df["Gender"].fillna("unknown").astype(str).str.lower().str.strip()
        df["GenderNum"] = df["Gender"].apply(lambda g: 1 if g == "male" else 0)

        df["Neighbourhood"] = df["Neighbourhood"].fillna("none").astype(str).str.lower().str.strip()
        df["Substance"] = df["Substance"].fillna("none").astype(str).str.lower().str.strip()

        df["OriginalSubstance"] = df["Substance"]
        df["OriginalNeighbourhood"] = df["Neighbourhood"]

        # 3. Target
        df["IsOverdose"] = df["Substance"].apply(classify_overdose_risk)
        df = df.dropna(subset=["AgeNumeric"])

        # 4. Age group feature
        df["AgeGroup"] = pd.cut(df["AgeNumeric"], bins=[0, 18, 25, 35, 50, 100], 
                               labels=["Under18", "YoungAdult", "Adult", "MiddleAge", "Senior"])

        # 5. One-hot encoding
        df_encoded = pd.get_dummies(df, columns=["AgeGroup", "Neighbourhood"], 
                                   prefix=["age", "neigh"], drop_first=False)

        # 6. Drop unused columns
        unwanted_columns = ["Neighbourhood ID", "Incident Number", "Dispatch Date", 
                            "Patient Number", "Ward", "Age", "Gender", "Substance",
                            "OriginalSubstance", "OriginalNeighbourhood"]
        df_encoded = df_encoded.drop(columns=[col for col in unwanted_columns if col in df_encoded.columns])

        # 7. Define features and target
        feature_cols = ["AgeNumeric", "GenderNum"] + \
                       [col for col in df_encoded.columns if col.startswith("age_") or col.startswith("neigh_")]
        X = df_encoded[feature_cols]
        y = df_encoded["IsOverdose"]

        # 8. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

        # 9. Base model
        base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight={0: 1, 1: 0.05, 2: 50}
        )
        base_model.fit(X_train, y_train)
        print("Base model trained.")

        # 10. Calibration
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        print("Calibrated model trained.")

        # 11. Save calibrated model + features
        joblib.dump(calibrated_model, "model_calibrated.pkl")
        joblib.dump(feature_cols, "feature_cols.pkl")
        print("Calibrated model and features saved.")

        # 12. Cross-validation (on uncalibrated model)
        cv_scores = cross_val_score(base_model, X_train, y_train, cv=5, scoring='f1_weighted')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f}")

        # 13. Feature importance
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": base_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        print("\nTop 10 Feature Importances:")
        print(importance_df.head(10))

        # 14. Evaluation
        y_pred = calibrated_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')

        # 15. Visualizations
        df_test_merged = pd.DataFrame({
            'AgeNumeric': X_test['AgeNumeric'],
            'GenderNum': X_test['GenderNum'],
            'PredictedRisk': y_pred,
            'ActualRisk': y_test
        })

        risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        df_test_merged['PredictedRiskCategory'] = df_test_merged['PredictedRisk'].map(risk_map)
        df_test_merged['ActualRiskCategory'] = df_test_merged['ActualRisk'].map(risk_map)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.boxplot(x='PredictedRiskCategory', y='AgeNumeric', data=df_test_merged)
        plt.title('Age vs Predicted Risk')

        plt.subplot(2, 2, 2)
        sns.countplot(x='ActualRiskCategory', hue='PredictedRiskCategory', data=df_test_merged)
        plt.title('Actual vs Predicted Risk')

        plt.subplot(2, 2, 3)
        top_features = importance_df.head(10)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 10 Feature Importance')

        plt.tight_layout()
        plt.savefig('model_analysis.png')
        print("Analysis visualizations saved.")

        return calibrated_model, feature_cols

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    model, features = train_and_save_model()