# supply_chain_risk_dataset.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# ============================
# Setup directories
# ============================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ============================
# Load dataset
# ============================
df = pd.read_csv("data/supply_chain_risk_dataset.csv")
print(df["manual_risk_label"].value_counts())

# Drop irrelevant columns
drop_cols = [
    "timestamp", "device_id", "order_id",
    "order_placed_date", "expected_delivery_date",
    "actual_delivery_date", "supplier_id", "system_log_message",
    "news_alert", "social_media_feed", "shipment_status"
]
df = df.drop(columns=drop_cols)

# ============================
# EDA
# ============================
eda_path = "results/supply_chain_risk_eda.txt"
with open(eda_path, "w") as f:
    f.write("===== EDA =====\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Label distribution:\n")
    f.write(str(df["manual_risk_label"].value_counts()) + "\n\n")
    f.write("Descriptive Statistics:\n")
    f.write(str(df.describe(include="all")) + "\n")
    eda_file = os.path.join("results", f"EDA_supply_chain_risk.csv")
    df.describe(include="all").transpose().to_csv(eda_file)

# Save descriptive stats to CSV
df.describe(include="all").to_csv("results/supply_chain_risk_descriptive_stats.csv")

# ============================
# Preprocessing
# ============================
y = df["manual_risk_label"]
X = df.drop(columns=["manual_risk_label"])

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# Models
# ============================
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
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(report + "\n")

        # ============================
        # Cross-validation
        # ============================
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f.write(f"Cross-Validation Accuracies: {cv_scores}\n")
        f.write(f"Mean CV Accuracy: {cv_scores.mean():.4f}\n")
        f.write("\n" + "-" * 60 + "\n")

        # ============================
        # Confusion matrices
        # ============================
        cm = confusion_matrix(y_test, preds, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix (Test) - {name}")
        plt.savefig(f"plots/supply_chain_risk_confmat_{name}.png")
        plt.close()

        # ============================
        # Feature importance
        # ============================
        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            feature_importances = np.abs(model.coef_).mean(axis=0)

        if feature_importances is not None:
            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": feature_importances
            }).sort_values(by="importance", ascending=False).head(10)

            f.write("\nTop 10 Features:\n")
            for _, row in importance_df.iterrows():
                f.write(f"{row['feature']}: {row['importance']:.4f}\n")

            # Plot feature importances
            plt.figure(figsize=(8, 6))
            sns.barplot(data=importance_df, x="importance", y="feature")
            plt.title(f"Top 10 Features - {name}")
            plt.savefig(f"plots/supply_chain_risk_feature_importance_{name}.png")
            plt.close()

        f.write("\n" + "-" * 60 + "\n")
