# supply_chain_risk_dataset_for_smes.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import numpy as np

# ============================
# Setup directories
# ============================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================
# Load dataset
# ============================
df = pd.read_csv("data/supply_chain_risk_dataset_for_smes.csv")

print(df["risk_label"].value_counts())

# Drop irrelevant / ID columns
drop_cols = ["timestamp", "machine_id", "supplier_id"]
df = df.drop(columns=drop_cols)

# ============================
# Classification Task (risk_label)
# ============================
y_class = df["risk_label"]
X_class = df.drop(columns=["risk_label", "risk_probability"])  # drop both targets

# Encode categorical variables
for col in X_class.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_class[col] = le.fit_transform(X_class[col])

# Encode target
if y_class.dtype == "object":
    le_y = LabelEncoder()
    y_class = le_y.fit_transform(y_class)

# Train-test split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Scale
scaler_class = StandardScaler()
Xc_train = scaler_class.fit_transform(Xc_train)
Xc_test = scaler_class.transform(Xc_test)

# Models for classification
classifiers = {
    "decision_tree_classifier": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "random_forest_classifier": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "logistic_regression_classifier": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
}

class_results_path = "results/supply_chain_risk_classification_results_for_smes.txt"
with open(class_results_path, "w") as f:
    f.write("===== Classification Results (risk_label) =====\n")

    for name, model in classifiers.items():
        model.fit(Xc_train, yc_train)
        preds = model.predict(Xc_test)

        acc = accuracy_score(yc_test, preds)
        report = classification_report(yc_test, preds)

        # Save model
        joblib.dump(model, f"models/supply_chain_risk_{name}_for_smes.pkl")

        # Write results
        f.write(f"\n{name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
        f.write("\n" + "-" * 60 + "\n")

        # Feature importance (tree-based models only)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # top 15
            f.write("\nTop 15 Features (by importance):\n")
            for idx in indices:
                f.write(f"{X_class.columns[idx]}: {importances[idx]:.4f}\n")
            f.write("\n" + "-" * 60 + "\n")

# ============================
# Regression Task (risk_probability)
# ============================
y_reg = df["risk_probability"]
X_reg = df.drop(columns=["risk_probability", "risk_label"])  # drop both targets

# Encode categorical
for col in X_reg.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_reg[col] = le.fit_transform(X_reg[col])

# Train-test split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scale
scaler_reg = StandardScaler()
Xr_train = scaler_reg.fit_transform(Xr_train)
Xr_test = scaler_reg.transform(Xr_test)

# Models for regression
regressors = {
    "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
    "random_forest_regressor": RandomForestRegressor(random_state=42),
    "linear_regression_regressor": LinearRegression()
}

reg_results_path = "results/supply_chain_risk_regression_results_for_smes.txt"
with open(reg_results_path, "w") as f:
    f.write("===== Regression Results (risk_probability) =====\n")

    for name, model in regressors.items():
        model.fit(Xr_train, yr_train)
        preds = model.predict(Xr_test)

        mse = mean_squared_error(yr_test, preds)
        r2 = r2_score(yr_test, preds)

        # Save model
        joblib.dump(model, f"models/supply_chain_risk_{name}_for_smes.pkl")

        # Write results
        f.write(f"\n{name}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write("\n" + "-" * 60 + "\n")

        # Feature importance (tree-based models only)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # top 15
            f.write("\nTop 15 Features (by importance):\n")
            for idx in indices:
                f.write(f"{X_reg.columns[idx]}: {importances[idx]:.4f}\n")
            f.write("\n" + "-" * 60 + "\n")
