
# ==========================================
# DIABETES RISK CLASSIFICATION
# IBM AI & ML INTERNSHIP PROJECT
# ==========================================

# 1. Import libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

print("DIABETES RISK CLASSIFICATION PROJECT")

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv("data/diabetes.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

print("\nFIRST FIVE RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

print("\nDIABETES DISTRIBUTION")
print(df["Outcome"].value_counts())

# ==========================================
# 4. ZERO VALUE ANALYSIS
# ==========================================

columns_to_check = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\nZERO VALUE ANALYSIS")

for column in columns_to_check:
    print(column, ":", (df[column] == 0).sum())

# ==========================================
# 5. DATA CLEANING
# ==========================================

# Remove duplicate records
df = df.drop_duplicates().copy()

# Replace medically invalid zero measurements
df[columns_to_check] = df[columns_to_check].replace(0, np.nan)

print("\nMISSING VALUES AFTER ZERO REPLACEMENT")
print(df.isnull().sum())

# ==========================================
# 6. FEATURE SELECTION
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFEATURES")
print(X.columns.tolist())

print("\nTARGET VARIABLE: Outcome")

# ==========================================
# 7. TRAIN AND TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# ==========================================
# 8. MACHINE LEARNING MODELS
# ==========================================

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])
}

results = {}

# ==========================================
# 9. MODEL TRAINING AND EVALUATION
# ==========================================

for name, model in models.items():

    print("\n================================")
    print("MODEL:", name)
    print("================================")

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))

# ==========================================
# 10. MODEL COMPARISON
# ==========================================

print("\nMODEL COMPARISON")

for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

# Select the model with highest test accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBEST MODEL:", best_model_name)

# ==========================================
# 11. SAVE TRAINED MODEL
# ==========================================

joblib.dump(best_model, "diabetes_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")
print("File: diabetes_model.pkl")

print("\nPROJECT EXECUTION COMPLETED!")
