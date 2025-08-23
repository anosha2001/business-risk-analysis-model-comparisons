# dynamic_supply_chain_logistics_dataset.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib

# ============================
# Setup directories
# ============================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================
# Load dataset
# ============================
df = pd.read_csv("data/dynamic_supply_chain_logistics_dataset.csv")

# Drop timestamp (not useful for modeling directly)
df = df.drop(columns=["timestamp"])

# ============================
# Classification Task: risk_classification
# ============================
X_cls = df.drop(columns=["risk_classification", "disruption_likelihood_score"])
y_cls = df["risk_classification"]

# Encode categorical features
for col in X_cls.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_cls[col] = le.fit_transform(X_cls[col])

# Encode target labels
le_y = LabelEncoder()
y_cls = le_y.fit_transform(y_cls)

# Train-test split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Scale features
scaler_cls = StandardScaler()
X_train_cls = scaler_cls.fit_transform(X_train_cls)
X_test_cls = scaler_cls.transform(X_test_cls)

# Models for classification
classifiers = {
    "decision_tree_classifier": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "random_forest_classifier": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "logistic_regression_classifier": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
}

cls_results_path = "results/dynamic_supply_chain_classification_results.txt"
with open(cls_results_path, "w") as f:
    f.write("===== Classification Results (risk_classification) =====\n")

    for name, model in classifiers.items():
        model.fit(X_train_cls, y_train_cls)
        preds = model.predict(X_test_cls)

        acc = accuracy_score(y_test_cls, preds)
        report = classification_report(y_test_cls, preds, target_names=le_y.classes_)

        # Save model
        joblib.dump(model, f"models/dynamic_supply_chain_classification_{name}.pkl")

        # Write results
        f.write(f"\n{name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)

        # Feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = None

        if importances is not None:
            feature_importance = sorted(zip(X_cls.columns, importances), key=lambda x: x[1], reverse=True)[:15]
            f.write("\nTop 15 Features:\n")
            for feat, score in feature_importance:
                f.write(f"{feat}: {score:.4f}\n")

        f.write("\n" + "-" * 60 + "\n")


# ============================
# Regression Task: disruption_likelihood_score
# ============================
X_reg = df.drop(columns=["risk_classification", "disruption_likelihood_score"])
y_reg = df["disruption_likelihood_score"]

# Encode categorical features
for col in X_reg.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_reg[col] = le.fit_transform(X_reg[col])

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scale features
scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# Models for regression
regressors = {
    "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
    "random_forest_regressor": RandomForestRegressor(random_state=42),
    "linear_regression_regressor": LinearRegression()
}

reg_results_path = "results/dynamic_supply_chain_regression_results.txt"
with open(reg_results_path, "w") as f:
    f.write("===== Regression Results (disruption_likelihood_score) =====\n")

    for name, model in regressors.items():
        model.fit(X_train_reg, y_train_reg)
        preds = model.predict(X_test_reg)

        mse = mean_squared_error(y_test_reg, preds)
        r2 = r2_score(y_test_reg, preds)

        # Save model
        joblib.dump(model, f"models/dynamic_supply_chain_regression_{name}.pkl")

        # Write results
        f.write(f"\n{name}\n")
        f.write(f"MSE: {mse:.4f}, R2: {r2:.4f}\n")

        # Feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = None

        if importances is not None:
            feature_importance = sorted(zip(X_reg.columns, importances), key=lambda x: x[1], reverse=True)[:15]
            f.write("\nTop 15 Features:\n")
            for feat, score in feature_importance:
                f.write(f"{feat}: {score:.4f}\n")

        f.write("-" * 60 + "\n")
