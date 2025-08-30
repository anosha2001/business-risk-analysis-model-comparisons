# dynamic_supply_chain_logistics_dataset.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
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
dataset_path = "data/dynamic_supply_chain_logistics_dataset.csv"
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
df = pd.read_csv(dataset_path)

# Drop timestamp (not useful for modeling directly)
df = df.drop(columns=["timestamp"])

# ============================
# EDA
# ============================
eda_path = f"results/{dataset_name}_EDA.txt"
with open(eda_path, "w") as f:
    f.write(f"===== EDA Report for {dataset_name} =====\n\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Data Types:\n")
    f.write(str(df.dtypes) + "\n\n")
    f.write("Descriptive Statistics:\n")
    eda_file = os.path.join("results", f"EDA_{dataset_name}.csv")
    df.describe(include="all").transpose().to_csv(eda_file)
    if "risk_classification" in df.columns:
        f.write("Risk Classification Distribution:\n")
        f.write(str(df["risk_classification"].value_counts()) + "\n\n")

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
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

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

cls_results_path = f"results/{dataset_name}_classification_results.txt"
with open(cls_results_path, "w") as f:
    f.write(f"===== Classification Results ({dataset_name}) =====\n")

    for name, model in classifiers.items():
        model.fit(X_train_cls, y_train_cls)
        preds = model.predict(X_test_cls)

        acc = accuracy_score(y_test_cls, preds)
        report = classification_report(y_test_cls, preds, target_names=le_y.classes_)

        # Save model
        joblib.dump(model, f"models/{dataset_name}_{name}.pkl")

        # Write results
        f.write(f"\n{name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)

        # Confusion matrix
        cm = confusion_matrix(y_test_cls, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name} ({dataset_name})")
        plt.savefig(f"plots/{dataset_name}_{name}_confusion_matrix.png")
        plt.close()

        # Feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = None

        if importances is not None:
            feature_importance = sorted(
                zip(X_cls.columns, importances), key=lambda x: x[1], reverse=True
            )[:15]
            f.write("\nTop 15 Features:\n")
            for feat, score in feature_importance:
                f.write(f"{feat}: {score:.4f}\n")

        f.write("\n" + "-" * 60 + "\n")

        # 5-fold cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_cls, y_cls, cv=cv, scoring="accuracy")
        f.write(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write("\n" + "="*60 + "\n")


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
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

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

reg_results_path = f"results/{dataset_name}_regression_results.txt"
with open(reg_results_path, "w") as f:
    f.write(f"===== Regression Results ({dataset_name}) =====\n")

    for name, model in regressors.items():
        model.fit(X_train_reg, y_train_reg)
        preds = model.predict(X_test_reg)

        mse = mean_squared_error(y_test_reg, preds)
        r2 = r2_score(y_test_reg, preds)

        # Save model
        joblib.dump(model, f"models/{dataset_name}_{name}.pkl")

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

        # 5-fold CV for regression
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_reg, y_reg, cv=cv, scoring="r2")
        f.write(f"5-Fold CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

        f.write("-" * 60 + "\n")
