from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def make_demo_data(n=2500, seed=42):
    """Create reproducible synthetic data for code demonstration, not business claims."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = np.clip(rng.normal(72, 25, n), 18, 160)
    support_calls = rng.poisson(1.2, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.58, .25, .17])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[.48, .37, .15])
    payment = rng.choice(["Electronic", "Card", "Bank transfer"], n)
    logit = (
        1.0
        + 0.035 * (monthly - 70)
        - 0.035 * (tenure - 24)
        + 0.42 * support_calls
        + 1.0 * (contract == "Month-to-month")
        - 0.65 * (contract == "Two year")
        + 0.4 * (internet == "Fiber")
        + 0.25 * (payment == "Electronic")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly.round(2),
        "support_calls": support_calls,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "churn": churn
    })

def main():
    df = make_demo_data()
    df.to_csv(OUT / "synthetic_customer_churn.csv", index=False)

    X = df.drop(columns="churn")
    y = df["churn"]
    numeric = ["tenure_months", "monthly_charges", "support_calls"]
    categorical = ["contract_type", "internet_service", "payment_method"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=15,
                                                class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42,
                                                n_jobs=-1)
    }

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (name, model) in zip(axes, models.items()):
        pipe = Pipeline([("preprocess", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob)
        }
        rows.append(row)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title(name, fontsize=9)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, zero_division=0))

    plt.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    print("\n=== Test-set model comparison (synthetic data) ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved outputs to: {OUT}")

if __name__ == "__main__":
    main()
