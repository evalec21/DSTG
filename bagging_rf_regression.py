import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`*")

# =============================
# CONFIG
# =============================
FAST_MODE = False  # True = brzo testiranje, False = finalni run

METRICS_CSV = "results_cv_metric.csv"
IMPORTANCE_CSV = "rf_permutation_importance.csv"
OUT_DIR = "figures"

TARGET_COL = "price"


# -----------------------------
# 1) Data
# -----------------------------
def load_data() -> pd.DataFrame:
    return sns.load_dataset("diamonds")


# -----------------------------
# 2) Preprocess
# -----------------------------
def build_preprocess_pipeline(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    return X, y, preprocessor, categorical_cols, numeric_cols


# -----------------------------
# 3) Metrics (RMSE, MAE, R2)
# -----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


rmse_scorer = make_scorer(rmse, greater_is_better=False)  # negative RMSE
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


# -----------------------------
# 4) CV eval + save
# -----------------------------
def evaluate_with_cv(model: Pipeline, X, y, n_splits: int = 5, random_state: int = 42, n_jobs: int = 1):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {"rmse": rmse_scorer, "mae": mae_scorer, "r2": "r2"}

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=n_jobs,
    )

    rmse_scores = -results["test_rmse"]
    mae_scores = -results["test_mae"]
    r2_scores = results["test_r2"]

    summary = {
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std(ddof=1)),
        "mae_mean": float(mae_scores.mean()),
        "mae_std": float(mae_scores.std(ddof=1)),
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std(ddof=1)),
    }

    return summary


def save_cv_summary(summary: dict, model_name: str, fast_mode: bool, filename: str = METRICS_CSV):
    new_row = {
        "model": model_name,
        "fast_mode": fast_mode,
        **summary,
    }

    if os.path.exists(filename):
        try:
            existing = pd.read_csv(filename)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=new_row.keys())
    else:
        existing = pd.DataFrame(columns=new_row.keys())

    updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    updated = updated.drop_duplicates(subset=["model", "fast_mode"], keep="last")
    updated.to_csv(filename, index=False)


# -----------------------------
# 5) Predicted vs Actual plots
# -----------------------------
def plot_predicted_vs_actual(y_true, y_pred, title: str, out_path: str):
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, alpha=0.25, s=10)

    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], linewidth=2)

    plt.title(title)
    plt.xlabel("Stvarna vrijednost (price)")
    plt.ylabel("Predviđena vrijednost (price)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def make_pred_vs_actual_plots(pipe: Pipeline, X, y, model_name: str, out_dir: str = OUT_DIR, random_state: int = 42):
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    # fit samo na train (korektno), predict train+test
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    plot_predicted_vs_actual(
        y_train,
        y_pred_train,
        title=f"{model_name}: Predikcije na trening skupu",
        out_path=os.path.join(out_dir, f"{model_name.lower()}_pred_vs_actual_train.png"),
    )

    plot_predicted_vs_actual(
        y_test,
        y_pred_test,
        title=f"{model_name}: Predikcije na testnom skupu",
        out_path=os.path.join(out_dir, f"{model_name.lower()}_pred_vs_actual_test.png"),
    )


# -----------------------------
# 6) Permutation importance (RF)
# -----------------------------
def compute_permutation_importance(fitted_pipeline: Pipeline, X, y, n_repeats: int = 10, random_state: int = 42):
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()

    X_transformed = preprocessor.transform(X)
    estimator = fitted_pipeline.named_steps["model"]

    perm = permutation_importance(
        estimator,
        X_transformed,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="r2",
        n_jobs=-1,
    )

    df = pd.DataFrame(
        {"feature": feature_names, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)

    return df


# -----------------------------
# 7) Plotting from CSVs
# -----------------------------
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_final_metrics():
    df = pd.read_csv(METRICS_CSV)
    df = df[df["fast_mode"] == False].copy()

    if df.empty:
        raise ValueError("Nema redova s fast_mode == False u results_cv_metric.csv")

    df["_row"] = range(len(df))
    df = df.sort_values("_row").groupby("model", as_index=False).tail(1).drop(columns=["_row"])

    preferred_order = ["DecisionTree", "Bagging", "RandomForest"]
    df["model"] = pd.Categorical(df["model"], categories=preferred_order, ordered=True)
    df = df.sort_values("model")
    return df


def plot_metrics_bar(df_metrics: pd.DataFrame):
    for metric, title, ylabel in [
        ("rmse_mean", "Usporedba modela (RMSE, 5-fold CV)", "RMSE (manje je bolje)"),
        ("mae_mean", "Usporedba modela (MAE, 5-fold CV)", "MAE (manje je bolje)"),
        ("r2_mean", "Usporedba modela (R², 5-fold CV)", "R² (više je bolje)"),
    ]:
        plt.figure()
        plt.bar(df_metrics["model"].astype(str), df_metrics[metric])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Model")

        for i, v in enumerate(df_metrics[metric].values):
            plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

        out_path = os.path.join(OUT_DIR, f"{metric}_bar.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")


def load_importances_top(n=10):
    df = pd.read_csv(IMPORTANCE_CSV)
    df = df[df["fast_mode"] == False].copy()

    if df.empty:
        raise ValueError("Nema redova s fast_mode == False u rf_permutation_importance.csv")

    return df.sort_values("importance_mean", ascending=False).head(n)


def plot_importances_bar(df_imp: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["feature"], df_imp["importance_mean"])
    plt.gca().invert_yaxis()
    plt.title("Top značajke prema permutation importance (R²)")
    plt.xlabel("Mean decrease in R² (više = važnije)")
    plt.ylabel("Značajka (feature)")

    out_path = os.path.join(OUT_DIR, "rf_permutation_importance_top10.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_importances_with_std(df_imp: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["feature"], df_imp["importance_mean"], xerr=df_imp["importance_std"])
    plt.gca().invert_yaxis()
    plt.title("Permutation importance + varijabilnost (std)")
    plt.xlabel("Mean decrease in R² (± std)")
    plt.ylabel("Značajka (feature)")

    out_path = os.path.join(OUT_DIR, "rf_permutation_importance_top10_std.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def compute_train_test_metrics(pipe: Pipeline, X, y, random_state: int = 42):
    """
    Fit na train, izračun RMSE/MAE/R2 na train i test skupu.
    Vraća (metrics_train, metrics_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    pipe.fit(X_train, y_train)

    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)

    train_metrics = {
        "rmse": rmse(y_train, pred_train),
        "mae": float(mean_absolute_error(y_train, pred_train)),
        "r2": float(pipe.score(X_train, y_train)),  # R2
    }

    test_metrics = {
        "rmse": rmse(y_test, pred_test),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "r2": float(pipe.score(X_test, y_test)),  # R2
    }

    return train_metrics, test_metrics


def save_train_test_metrics(rows: list[dict], filename: str = "results_train_test_metrics.csv"):
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

def plot_train_test_metrics(filename="results_train_test_metrics.csv", out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(filename)

    for metric, title, ylabel in [
        ("rmse", "RMSE (train vs test)", "RMSE (manje je bolje)"),
        ("mae", "MAE (train vs test)", "MAE (manje je bolje)"),
        ("r2", "R² (train vs test)", "R² (više je bolje)"),
    ]:
        plt.figure()
        # stupci: model_train, model_test
        labels = [f"{m}-{s}" for m, s in zip(df["model"], df["split"])]
        plt.bar(labels, df[metric])
        plt.xticks(rotation=20, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"train_test_{metric}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")

def plot_pdp_carat(
    fitted_pipe: Pipeline,
    X,
    out_dir: str = "figures",
):
    """
    PDP za varijablu 'carat' na Random Forest modelu.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    PartialDependenceDisplay.from_estimator(
        fitted_pipe,
        X,
        features=["carat"],
        grid_resolution=50,
        kind="average",
    )

    plt.title("Partial Dependence Plot (PDP) za varijablu 'carat' – Random Forest")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "rf_pdp_carat.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_out_dir()

    if FAST_MODE:
        rf_estimators = 80
        bag_estimators = 50
        cv_splits = 3
        do_grid_search = False
        perm_repeats = 5
        cv_n_jobs = 1
        print("=== FAST_MODE: ON (brzo testiranje) ===")
    else:
        rf_estimators = 300
        bag_estimators = 300
        cv_splits = 5
        do_grid_search = True
        perm_repeats = 10
        cv_n_jobs = 1
        print("=== FAST_MODE: OFF (finalni run) ===")

    df = load_data()
    X, y, preprocessor, cat_cols, num_cols = build_preprocess_pipeline(df, TARGET_COL)

    print("Dataset shape:", df.shape)
    print("Categorical columns:", cat_cols)
    print("Numeric columns:", num_cols)

    # --- RF baseline ---
    rf = RandomForestRegressor(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
    rf_pipe = Pipeline([("preprocess", preprocessor), ("model", rf)])

    rf_summary = evaluate_with_cv(rf_pipe, X, y, n_splits=cv_splits, n_jobs=cv_n_jobs)
    print("\nRandomForest CV:", rf_summary)
    save_cv_summary(rf_summary, "RandomForest", FAST_MODE)

    # --- Bagging baseline ---
    base_tree = DecisionTreeRegressor(random_state=42)
    bag = BaggingRegressor(estimator=base_tree, n_estimators=bag_estimators, random_state=42, n_jobs=-1)
    bag_pipe = Pipeline([("preprocess", preprocessor), ("model", bag)])

    bag_summary = evaluate_with_cv(bag_pipe, X, y, n_splits=cv_splits, n_jobs=cv_n_jobs)
    print("\nBagging CV:", bag_summary)
    save_cv_summary(bag_summary, "Bagging", FAST_MODE)

    # --- RF tuning (final only) ---
    best_rf_pipe = rf_pipe
    if do_grid_search:
        rf_tune = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_tune_pipe = Pipeline([("preprocess", preprocessor), ("model", rf_tune)])

        param_grid = {
            "model__n_estimators": [200, 500],
            "model__max_features": ["sqrt", 0.5],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }

        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=rf_tune_pipe,
            param_grid=param_grid,
            scoring=rmse_scorer,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X, y)
        print("\nRF tuning best RMSE:", round(-grid.best_score_, 4))
        print("RF tuning best params:", grid.best_params_)
        best_rf_pipe = grid.best_estimator_

    # --- Predicted vs Actual (train/test) ---
    make_pred_vs_actual_plots(best_rf_pipe, X, y, model_name="RandomForest")
    make_pred_vs_actual_plots(bag_pipe, X, y, model_name="Bagging")

    # -------------------------
    # Train/Test metrike (kao u literaturi)
    # -------------------------
    rows = []

    rf_train, rf_test = compute_train_test_metrics(best_rf_pipe, X, y, random_state=42)
    rows.append({"model": "RandomForest", "split": "train", **rf_train})
    rows.append({"model": "RandomForest", "split": "test", **rf_test})

    bag_train, bag_test = compute_train_test_metrics(bag_pipe, X, y, random_state=42)
    rows.append({"model": "Bagging", "split": "train", **bag_train})
    rows.append({"model": "Bagging", "split": "test", **bag_test})

    save_train_test_metrics(rows, filename="results_train_test_metrics.csv")
    plot_train_test_metrics()

    # print tablica u konzolu (zgodno za copy u dokumentaciju)
    df_tt = pd.DataFrame(rows)
    print("\nTrain/Test metrike:")
    print(df_tt.to_string(index=False))


    # --- Permutation importance (fit on full data) ---
    best_rf_pipe.fit(X, y)
    imp_df = compute_permutation_importance(best_rf_pipe, X, y, n_repeats=perm_repeats, random_state=42)
    imp_df["fast_mode"] = FAST_MODE
    imp_df.to_csv(IMPORTANCE_CSV, index=False)

    # -------------------------
    # PDP za varijablu 'carat' (Random Forest)
    # -------------------------
    plot_pdp_carat(best_rf_pipe, X, out_dir=OUT_DIR)


    print("\nTop 10 permutation importances (R2):")
    print(imp_df.head(10).to_string(index=False))

    # --- Generate bar charts from CSVs ---
    if not FAST_MODE:
        df_metrics = load_final_metrics()
        plot_metrics_bar(df_metrics)

        df_imp_top10 = load_importances_top(10)
        plot_importances_bar(df_imp_top10)
        plot_importances_with_std(df_imp_top10)

    print("\nGotovo. Grafovi su u folderu:", OUT_DIR)


if __name__ == "__main__":
    main()
