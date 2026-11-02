import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# --- DYNAMIC PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points exactly to the 'saved_models' folder shown in your image
model_dir = os.path.join(BASE_DIR, "saved_models")

subjects = ["Math", "Science", "English", "Social"]
subject_models = {}
label_enc = LabelEncoder()

# Fit the encoder for parent occupation
parent_occupations = ["Clerk", "Business", "Artist", "Doctor", "Teacher", "Engineer"]
label_enc.fit(parent_occupations)

# Load the saved models from the 'saved_models' folder
for subject in subjects:
    model_path = os.path.join(model_dir, f"{subject.lower()}_model.pkl")
    
    try:
        with open(model_path, "rb") as file:
            subject_models[subject] = pickle.load(file)
        print(f"Successfully loaded model for {subject} from {model_path}")
    except Exception as e:
        print(f"Failed to load model for {subject}: {e}")

# Classification function based on assessment score
def classify_level(score):
    if score >= 80: return "Advanced"
    elif score >= 60: return "Intermediate"
    elif score >= 40: return "Beginner"
    else: return "Needs Improvement"

def predict_assessment_scores(student_data):
    common_features = ["Age", "Grade", "IQ Level", "Parent Occupation", "Study Time (hrs/week)"]
    input_data = {
        "Age": student_data["age"],
        "Grade": student_data["grade"],
        "IQ Level": 100, 
        "Parent Occupation": label_enc.transform([student_data["parent_occupation"]])[0],
        "Study Time (hrs/week)": student_data["study_time"]
    }

    results = {}
    for subject in subjects:
        subject_score_key = f"{subject.lower()}_score"
        input_df = pd.DataFrame([input_data])
        input_df[f"{subject} Score"] = student_data[subject_score_key]
        features = common_features + [f"{subject} Score"]

        if subject in subject_models:
            try:
                predicted_score = subject_models[subject].predict(input_df[features])[0]
                results[subject] = {
                    "assessment_score": round(predicted_score, 2),
                    "level": classify_level(predicted_score)
                }
            except Exception as e:
                results[subject] = {"error": "Prediction failed"}
        else:
            results[subject] = {"error": "Model not loaded"}
    return results