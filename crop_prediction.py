import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# LOAD DATASET FROM GITHUB RAW URL
# ==========================================

url = "https://raw.githubusercontent.com/YOUR_USERNAME/upskillCampus/main/agriculture_crop_production.csv"

df = pd.read_csv(url)

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)
print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================================
# DATA CLEANING
# ==========================================

df = df.drop_duplicates()

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = label_encoder.fit_transform(df[col])

# ==========================================
# TARGET COLUMN
# ==========================================

target_column = "Production"

X = df.drop(target_column, axis=1)
y = df[target_column]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("=" * 50)
print("Mean Absolute Error :", round(mae, 2))
print("Mean Squared Error  :", round(mse, 2))
print("R2 Score            :", round(r2, 4))

# ==========================================
# FEATURE IMPORTANCE GRAPH
# ==========================================

importance = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Production")
plt.savefig("actual_vs_predicted.png")
plt.show()

# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv("prediction_results.csv", index=False)

print("\nFiles Generated:")
print("1. feature_importance.png")
print("2. actual_vs_predicted.png")
print("3. prediction_results.csv")
