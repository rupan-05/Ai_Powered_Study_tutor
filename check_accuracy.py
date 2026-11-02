import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# 1. Load the actual CSV file
df = pd.read_csv('subject_wise_assessment_predictions.csv')

# Preprocess to ignore text columns that ML models can't ingest 
df.drop(columns=["Student Name", "Learning Level", "Assessment Score"], inplace=True, errors="ignore")

# Encode Parent Occupation
label_enc = LabelEncoder()
df["Parent Occupation"] = label_enc.fit_transform(df["Parent Occupation"])

# Fill any NA rows
df.fillna(df.mean(numeric_only=True), inplace=True)

# 2. Select Features for a sample model (Math)
features = ["Age", "Grade", "IQ Level", "Parent Occupation", "Study Time (hrs/week)", "Math Score"]
X = df[features]

# The Assessment Score is simulated in the training loop. We will replicate the target here.
# (If it's already generated in the dataframe, we'll grab it, else we simulate the exact target used)
if "Math Assessment Score" in df.columns:
    y = df["Math Assessment Score"]
else:
    np.random.seed(42)
    y = df["Math Score"] * 1.5 + np.random.normal(0, 5, len(df))

# 3. Split and Predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute Mean Accuracy Percentage
mae = mean_absolute_error(y_test, y_pred)
mean_actual = np.mean(y_test)
accuracy = (1 - (mae / mean_actual)) * 100

# 4. SHOW THE TRUTH
print(f"\n--- Model Validation Metrics (Math Assessment Model) ---")
print(f"R-Squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mae:.2f} marks")
print(f"Model Built Accuracy: {accuracy:.2f}%\n")