import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\USER\OneDrive\Documents\project\subject_wise_assessment_predictions.csv")

# Step 1: Preprocessing
# Drop unnecessary columns (Student Name, Learning Level, and overall Assessment Score)
df.drop(columns=["Student Name", "Learning Level", "Assessment Score"], inplace=True, errors="ignore")

# Simulate subject-specific assessment scores (placeholder)
# In a real scenario, replace this with actual assessment scores
np.random.seed(42)  # For reproducibility
subjects = ["Math Score", "Science Score", "English Score", "Social Score"]
for subject in subjects:
    subject_name = subject.split()[0]  # e.g., "Math" from "Math Score"
    # Simulate assessment score as 1.5 * subject score + random noise
    df[f"{subject_name} Assessment Score"] = df[subject] * 1.5 + np.random.normal(0, 5, len(df))

# Encode categorical features
label_enc = LabelEncoder()
df["Parent Occupation"] = label_enc.fit_transform(df["Parent Occupation"])

# Check for missing values and handle them (if any)
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)  # Simple imputation with mean
    print("Missing values filled with mean.")
else:
    print("No missing values found.")

# Step 2: Feature Selection
# Features for each subject: Subject Score + Age, Grade, IQ Level, Parent Occupation, Study Time
common_features = ["Age", "Grade", "IQ Level", "Parent Occupation", "Study Time (hrs/week)"]
subject_models = {}

# Prepare a single X with all features (including all subject scores) for train-test split
all_features = common_features + subjects  # Include all subject scores
X = df[all_features]
# We’ll split the data once, but we’ll use different targets (y) for each subject
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Step 3: Model Building and Evaluation
# Define the save directory
save_dir = r"C:\Users\USER\OneDrive\Documents\project\saved_models"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn’t exist

for subject in subjects:
    subject_name = subject.split()[0]  # e.g., "Math" from "Math Score"
    print(f"\nTraining model for {subject_name} Assessment Score...")

    # Define features (X) and target (y) for the current subject
    features = common_features + [subject]  # Include the subject score as a feature
    y = df[f"{subject_name} Assessment Score"]

    # Split the target (y) to match X_train and X_test
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

    # Initialize XGBoost regressor
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    # Train the model
    xgb_model.fit(X_train[features], y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test[features])

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Compute accuracy using Mean Accuracy Percentage (MAP)
    mean_actual = np.mean(y_test)
    accuracy = (1 - (mae / mean_actual)) * 100

    print(f"📊 Model Performance for {subject_name} Assessment Score:")
    print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
    print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
    print(f"🔹 R² Score: {r2:.2f}")
    print(f"✅ Model Accuracy: {accuracy:.2f}%")

    # Save the model for this subject
    save_path = os.path.join(save_dir, f"xgb_model_{subject_name}_assessment.pkl")
    try:
        with open(save_path, "wb") as file:
            pickle.dump(xgb_model, file)
        print(f"✅ Model for {subject_name} Assessment Score saved at: {save_path}")
    except Exception as e:
        print(f"❌ Failed to save model for {subject_name}: {e}")

    # Store the model for later use
    subject_models[subject_name] = xgb_model

# Step 4: Classification
# Classification function based on assessment score
def classify_level(score):
    if score >= 80:
        return "Advanced"
    elif score >= 60:
        return "Intermediate"
    elif score >= 40:
        return "Beginner"
    else:
        return "Needs Improvement"

# Predict subject-wise assessment scores and classify levels for the test set
X_test = X_test.copy()
for subject in subjects:
    subject_name = subject.split()[0]
    model = subject_models[subject_name]

    # Ensure the correct feature list is used
    features = common_features + [subject]  # Include the correct subject score
    
    X_test[f"Predicted {subject_name} Assessment Score"] = model.predict(X_test[features])
    X_test[f"{subject_name} Level"] = X_test[f"Predicted {subject_name} Assessment Score"].apply(classify_level)

# Derive an overall learning level
# Method: Take the most frequent level across subjects
def get_overall_level(row):
    levels = [row[f"{subject.split()[0]} Level"] for subject in subjects]
    level_counts = pd.Series(levels).value_counts()
    return level_counts.index[0]  # Return the most frequent level

X_test["Overall Level"] = X_test.apply(get_overall_level, axis=1)

# Step 5: Display Results
print("\n🔍 Sample Classified Data:")
print(X_test[[f"Predicted {subject.split()[0]} Assessment Score" for subject in subjects] + 
             [f"{subject.split()[0]} Level" for subject in subjects] + ["Overall Level"]].head(10))

# Save test set with predictions and classifications
X_test.to_csv(r"C:\Users\USER\OneDrive\Documents\project\subject_wise_assessment_predictions.csv", index=False)
print("✅ Test set with subject-wise assessment predictions and classifications saved.")
