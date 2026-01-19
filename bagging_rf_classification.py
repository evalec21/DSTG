import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# =============================
# CONFIG
# =============================
FAST_MODE = True  # True = brzo testiranje, False = finalni run

TARGET_COL = "cut"
OUT_DIR = "figures"

CLS_CV_CSV = "results_cv_classification.csv"
CLS_TT_CSV = "results_train_test_classification.csv"
CLS_IMPORTANCE_CSV = "rf_permutation_importance_classification.csv"


# -----------------------------
# Utils
# -----------------------------
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# 1) Load data
# -----------------------------
def load_data():
    return sns.load_dataset("diamonds")


# -----------------------------
# 2) Preprocessing
# -----------------------------
def build_preprocess_pipeline(df: pd.DataFrame):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL, "price"])  # ne koristimo price u klasifikaciji

    categorical_cols = ["color", "clarity"]
    numeric_cols = ["carat", "depth", "table", "x", "y", "z"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    return X, y, preprocessor


# -----------------------------
# 3) CV evaluation
# -----------------------------
def evaluate_with_cv(pipe: Pipeline, X, y, fast_mode: bool):
    cv = StratifiedKFold(
        n_splits=3 if fast_mode else 5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(ddof=1),
        "f1_macro_mean": scores["test_f1_macro"].mean(),
        "f1_macro_std": scores["test_f1_macro"].std(ddof=1),
    }


def save_cv_summary(model_name: str, summary: dict, fast_mode: bool):
    row = {
        "model": model_name,
        "fast_mode": fast_mode,
        **summary,
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(CLS_CV_CSV):
        df_old = pd.read_csv(CLS_CV_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["model", "fast_mode"], keep="last")
    else:
        df = df_new

    df.to_csv(CLS_CV_CSV, index=False)


# -----------------------------
# 4) Train/Test metrics
# -----------------------------
def compute_train_test_metrics(pipe: Pipeline, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipe.fit(X_train, y_train)

    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)

    return [
        {
            "model": pipe.named_steps["model"].__class__.__name__,
            "split": "train",
            "accuracy": accuracy_score(y_train, pred_train),
            "f1_macro": f1_score(y_train, pred_train, average="macro"),
        },
        {
            "model": pipe.named_steps["model"].__class__.__name__,
            "split": "test",
            "accuracy": accuracy_score(y_test, pred_test),
            "f1_macro": f1_score(y_test, pred_test, average="macro"),
        },
    ], X_test, y_test


# -----------------------------
# 5) Confusion matrix
# -----------------------------
def plot_confusion_matrix(pipe: Pipeline, X_test, y_test, model_name: str):
    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=pipe.named_steps["model"].classes_)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=pipe.named_steps["model"].classes_,
    )

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix – {model_name}")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------
# 6) Permutation importance (RF)
# -----------------------------
def compute_permutation_importance_cls(pipe: Pipeline, X_test, y_test):
    preprocessor = pipe.named_steps["preprocess"]
    estimator = pipe.named_steps["model"]

    X_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    perm = permutation_importance(
        estimator,
        X_trans,
        y_test,
        n_repeats=10 if not FAST_MODE else 5,
        random_state=42,
        scoring="f1_macro",
        n_jobs=-1,
    )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    df["fast_mode"] = FAST_MODE
    df.to_csv(CLS_IMPORTANCE_CSV, index=False)
    return df


# -----------------------------
# 7) PDP for carat (RF)
# -----------------------------
def plot_pdp_carat(pipe: Pipeline, X, target_class: str = "Ideal"):
    """
    PDP za varijablu 'carat' u multi-class klasifikaciji.
    Prikazuje kako 'carat' utječe na vjerojatnost odabrane klase (npr. 'Ideal').

    Važno: target se prosljeđuje kao naziv klase (string), jer su classes_ stringovi.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = list(pipe.named_steps["model"].classes_)
    if target_class not in classes:
        raise ValueError(f"Klasa '{target_class}' nije u klasama modela: {classes}")

    plt.figure(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(
        pipe,
        X,
        features=["carat"],
        target=target_class,      # <-- KLJUČ: string, ne indeks
        kind="average",
        grid_resolution=50,
    )

    plt.title(f"PDP za 'carat' – P({target_class}) (Random Forest)")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"rf_pdp_carat_{target_class}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_out_dir()

    n_estimators = 100 if FAST_MODE else 400

    df = load_data()
    X, y, preprocessor = build_preprocess_pipeline(df)

    # -------------------------
    # MODELS (tvoj dio)
    # -------------------------
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    bag_pipe = Pipeline([("preprocess", preprocessor), ("model", bagging)])
    rf_pipe = Pipeline([("preprocess", preprocessor), ("model", rf)])

    # -------------------------
    # CV evaluation
    # -------------------------
    bag_cv = evaluate_with_cv(bag_pipe, X, y, FAST_MODE)
    rf_cv = evaluate_with_cv(rf_pipe, X, y, FAST_MODE)

    save_cv_summary("Bagging_Classifier", bag_cv, FAST_MODE)
    save_cv_summary("RandomForest_Classifier", rf_cv, FAST_MODE)

    print("\nCV Bagging:", bag_cv)
    print("CV RandomForest:", rf_cv)

    # -------------------------
    # Train/Test + confusion matrix
    # -------------------------
    rows = []

    bag_rows, X_test, y_test = compute_train_test_metrics(bag_pipe, X, y)
    rows.extend(bag_rows)
    plot_confusion_matrix(bag_pipe, X_test, y_test, "Bagging")

    rf_rows, X_test, y_test = compute_train_test_metrics(rf_pipe, X, y)
    rows.extend(rf_rows)
    plot_confusion_matrix(rf_pipe, X_test, y_test, "RandomForest")

    df_tt = pd.DataFrame(rows)
    df_tt.to_csv(CLS_TT_CSV, index=False)

    print("\nTrain/Test metrike:")
    print(df_tt.to_string(index=False))

    # -------------------------
    # Permutation importance + PDP (RF)
    # -------------------------
    rf_pipe.fit(X, y)
    imp_df = compute_permutation_importance_cls(rf_pipe, X_test, y_test)
    print("\nTop 10 permutation importance (classification):")
    print(imp_df.head(10).to_string(index=False))

    plot_pdp_carat(rf_pipe, X, target_class="Ideal")
    plot_pdp_carat(rf_pipe, X, target_class="Premium")

    print("\nGotovo. Svi rezultati su generirani.")


if __name__ == "__main__":
    main()
