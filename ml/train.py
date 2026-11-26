"""
Train heart-disease classifiers, evaluate them honestly, and export the best
model as JSON so the web app can run predictions fully in the browser (offline).

Usage:
    python ml/train.py                      # uses ml/dataset.csv
    python ml/train.py path/to/data.csv     # use any CSV with the same columns

Outputs:
    public/model.json    - scaler + logistic-regression weights used by the app
    public/metrics.json  - evaluation metrics for every model (shown in the app)
"""
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "ml" / "dataset.csv"
OUT = ROOT / "public"
SEED = 42

FEATURES = ["age", "sex", "blood_pressure", "cholesterol", "bmi", "smoker",
            "diabetes", "physical_activity", "family_history", "stress_level"]
TARGET = "heart_disease"


def load(path):
    df = pd.read_csv(path)
    df["sex"] = (df["sex"] == "Male").astype(int)
    for c in ("smoker", "diabetes", "family_history"):
        df[c] = (df[c] == "Yes").astype(int)
    return df[FEATURES], df[TARGET]


def scores(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    X, y = load(DATA)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        "Random Forest": RandomForestClassifier(300, max_depth=6, min_samples_leaf=5, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(max_depth=2, n_estimators=100,
                                                        learning_rate=0.05, random_state=SEED),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    results = []
    for name, model in models.items():
        cvs = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        model.fit(X_tr, y_tr)
        pred, prob = model.predict(X_te), model.predict_proba(X_te)[:, 1]
        results.append({
            "model": name,
            "test": scores(y_te, pred, prob),
            "cv_accuracy_mean": float(cvs["test_accuracy"].mean()),
            "cv_accuracy_std": float(cvs["test_accuracy"].std()),
            "cv_roc_auc_mean": float(cvs["test_roc_auc"].mean()),
            "train_accuracy": float(model.score(X_tr, y_tr)),
        })
        r = results[-1]
        print(f"{name:20s} CV acc {r['cv_accuracy_mean']:.3f} ± {r['cv_accuracy_std']:.3f} | "
              f"CV AUC {r['cv_roc_auc_mean']:.3f} | test acc {r['test']['accuracy']:.3f} | "
              f"train acc {r['train_accuracy']:.3f}")

    # Select by cross-validated AUC (robust on small data), not by the test set.
    best = max(results, key=lambda r: r["cv_roc_auc_mean"])
    print("Selected:", best["model"])

    # The web app runs a logistic regression in the browser. Fit it on all data
    # and export its parameters, regardless of which model scored best, so the
    # deployed app is small, explainable and works offline.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    scaler, clf = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]

    baseline = float(max(y.mean(), 1 - y.mean()))
    OUT.mkdir(exist_ok=True)
    (OUT / "model.json").write_text(json.dumps({
        "type": "logistic_regression",
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }, indent=2))

    best_cv_acc = max(r["cv_accuracy_mean"] for r in results)
    (OUT / "metrics.json").write_text(json.dumps({
        "trained_on": date.today().isoformat(),
        "n_samples": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "positive_rate": float(y.mean()),
        "majority_baseline_accuracy": baseline,
        "best_cv_model": best["model"],
        "deployed_model": "Logistic Regression",
        "meets_80_percent": bool(best_cv_acc >= 0.80),
        "best_cv_accuracy": float(best_cv_acc),
        "models": results,
    }, indent=2))
    print(f"Majority-class baseline: {baseline:.3f} | best CV accuracy: {best_cv_acc:.3f}")
    print("Wrote public/model.json and public/metrics.json")


if __name__ == "__main__":
    main()
