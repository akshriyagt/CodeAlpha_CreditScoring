"""
CodeAlpha ML Internship - Task 1: Credit Scoring Model
Predicts creditworthiness (0 = good, 1 = risky/default) from financial history.

Usage:
    python credit_scoring.py                      # runs on synthetic demo data
    python credit_scoring.py --data cs-training.csv --target SeriousDlqin2yrs
    (any CSV with a binary target column works)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)


def make_synthetic_data(n=5000, seed=42):
    """Demo data so the project runs without downloads. Replace with a real dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(21, 70, n),
        "income": rng.lognormal(10.8, 0.5, n).round(0),
        "total_debt": rng.lognormal(9.5, 0.9, n).round(0),
        "num_credit_lines": rng.integers(1, 15, n),
        "late_payments_12m": rng.poisson(0.8, n),
        "credit_history_years": rng.integers(0, 30, n),
        "credit_utilization": rng.beta(2, 4, n).round(3),
        "employment_years": rng.integers(0, 35, n),
    })
    z = (-3.0 + 0.9 * df.late_payments_12m + 2.0 * df.credit_utilization
         + 1.5 * (df.total_debt / df.income) - 0.05 * df.credit_history_years
         - 0.02 * df.employment_years + rng.normal(0, 0.7, n))
    df["default"] = (1 / (1 + np.exp(-z)) > rng.random(n)).astype(int)
    return df


def engineer_features(df, target):
    """Feature engineering from financial history."""
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    if "income" in cols and "total_debt" in cols:
        df["debt_to_income"] = df[cols["total_debt"]] / (df[cols["income"]] + 1)
    if "late_payments_12m" in cols and "num_credit_lines" in cols:
        df["late_per_line"] = df[cols["late_payments_12m"]] / (df[cols["num_credit_lines"]] + 1)
    if "credit_history_years" in cols and "age" in cols:
        df["history_to_age"] = df[cols["credit_history_years"]] / df[cols["age"]]
    return df


def main(args):
    df = pd.read_csv(args.data) if args.data else make_synthetic_data()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    target = args.target
    print(f"Dataset shape: {df.shape}\nClass balance:\n{df[target].value_counts(normalize=True).round(3)}\n")

    df = engineer_features(df, target)
    X, y = df.drop(columns=[target]), df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": Pipeline([("imp", SimpleImputer(strategy="median")),
                                         ("sc", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Decision Tree": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("m", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                                n_jobs=-1, random_state=42))]),
    }

    results, fitted = [], {}
    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
        results.append({"Model": name,
                        "Precision": precision_score(y_te, pred),
                        "Recall": recall_score(y_te, pred),
                        "F1-Score": f1_score(y_te, pred),
                        "ROC-AUC": roc_auc_score(y_te, proba)})
        fpr, tpr, _ = roc_curve(y_te, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[-1]['ROC-AUC']:.3f})")
        fitted[name] = model

    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Credit Scoring"); plt.legend(); plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150); plt.close()

    res = pd.DataFrame(results).set_index("Model").round(4)
    print("=== Model comparison ===\n", res, "\n")
    res.to_csv("results.csv")

    best = res["ROC-AUC"].idxmax()
    print(f"Best model by ROC-AUC: {best}")
    ConfusionMatrixDisplay(confusion_matrix(y_te, fitted[best].predict(X_te)),
                           display_labels=["Good", "Risky"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best}"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150); plt.close()

    rf = fitted["Random Forest"].named_steps["m"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    imp.plot(kind="barh", figsize=(7, 5), title="Random Forest Feature Importance")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="Path to CSV dataset")
    p.add_argument("--target", default="default", help="Name of binary target column")
    main(p.parse_args())
