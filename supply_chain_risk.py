# supply_chain_risk_dataset.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# ============================
# Setup directories
# ============================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================
# Load dataset
# ============================
df = pd.read_csv("data/supply_chain_risk_dataset.csv")
print(df["manual_risk_label"].value_counts())
# Drop irrelevant columns (IDs, timestamps, free text logs that won't help ML directly)
drop_cols = [
    "timestamp", "device_id", "order_id",
    "order_placed_date", "expected_delivery_date",
    "actual_delivery_date", "supplier_id", "system_log_message",
    "news_alert", "social_media_feed", "shipment_status"
]
df = df.drop(columns=drop_cols)

# Target variable
y = df["manual_risk_label"]
X = df.drop(columns=["manual_risk_label"])

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models for classification (with class weights)
classifiers = {
    "decision_tree_classifier": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "random_forest_classifier": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "logistic_regression_classifier": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
}

results_path = "results/supply_chain_risk_results.txt"
with open(results_path, "w") as f:
    f.write("===== Classification Results (manual_risk_label) =====\n")

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        # Save model
        joblib.dump(model, f"models/supply_chain_risk_{name}.pkl")

        # Write results
        f.write(f"\n{name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
        f.write("\n" + "-" * 60 + "\n")

        # Feature importance / coefficients
        f.write("\nTop 10 Features:\n")
        feature_importances = None
        if hasattr(model, "feature_importances_"):  # Tree-based models
            feature_importances = model.feature_importances_
        elif hasattr(model, "coef_"):  # Logistic regression
            feature_importances = np.abs(model.coef_).mean(axis=0)

        if feature_importances is not None:
            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": feature_importances
            }).sort_values(by="importance", ascending=False).head(10)

            for _, row in importance_df.iterrows():
                f.write(f"{row['feature']}: {row['importance']:.4f}\n")

        f.write("\n" + "-" * 60 + "\n")