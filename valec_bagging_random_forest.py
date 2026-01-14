import numpy as np
import pandas as pd
import seaborn as sns
import os
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`*")

# =============================
# DEVELOPMENT MODE SWITCH
# =============================
FAST_MODE = False  # True = brzo testiranje, False = finalni run


# -----------------------------
# 1) Učitavanje podataka (seaborn)
# -----------------------------
def load_data() -> pd.DataFrame:
    df = sns.load_dataset("diamonds")
    return df


# -----------------------------
# 2) Priprema X, y + preprocessing pipeline (one-hot)
#    (Scaling nije potreban za modele temeljene na stablima.)
# -----------------------------
def build_preprocess_pipeline(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}"
        )

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
# 3) Metrike (RMSE, MAE, R2)
# -----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


rmse_scorer = make_scorer(rmse, greater_is_better=False)  # negativno (sklearn očekuje "više je bolje")
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


# -----------------------------
# 4) Evaluacija modela kroz k-fold CV
# -----------------------------
def evaluate_with_cv(model: Pipeline, X, y, n_splits: int = 5, random_state: int = 42, n_jobs: int = 1):
    """
    Napomena:
    - Kod ansambl modela (posebno Bagging + RF) često je bolje da ovdje bude n_jobs=1,
      jer modeli sami koriste paralelizaciju (n_jobs=-1) i izbjegava se "nested parallelism".
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "rmse": rmse_scorer,
        "mae": mae_scorer,
        "r2": "r2",
    }

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

    return summary, results

def save_cv_summary(summary: dict, model_name: str, fast_mode: bool, filename: str = "results_cv_metric.csv"):
    new_row = {
        "model": model_name,
        "fast_mode": fast_mode,
        "rmse_mean": summary["rmse_mean"],
        "rmse_std": summary["rmse_std"],
        "mae_mean": summary["mae_mean"],
        "mae_std": summary["mae_std"],
        "r2_mean": summary["r2_mean"],
        "r2_std": summary["r2_std"],
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
# 5) Permutation feature importance (za Random Forest)
# -----------------------------
def compute_permutation_importance(
    fitted_pipeline: Pipeline, X, y, n_repeats: int = 10, random_state: int = 42
):
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

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df


# -----------------------------
# 6) Glavni dio: RF + Bagging + CV + tuning
# -----------------------------
def main():
    TARGET_COL = "price"

    # -------------------------
    # FAST / FINAL konfiguracija
    # -------------------------
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
        cv_n_jobs = 1  # preporuka: 1 zbog izbjegavanja nested paralelizacije
        print("=== FAST_MODE: OFF (finalni run) ===")

    df = load_data()

    # Brza provjera da je stvarno diamonds i da ima očekivanu veličinu
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    X, y, preprocessor, cat_cols, num_cols = build_preprocess_pipeline(df, TARGET_COL)

    print("Categorical columns:", cat_cols)
    print("Numeric columns:", num_cols)

    # -------------------------
    # Random Forest baseline
    # -------------------------
    rf = RandomForestRegressor(
        n_estimators=rf_estimators,
        random_state=42,
        n_jobs=-1,
    )

    rf_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf),
        ]
    )

    rf_cv_summary, _ = evaluate_with_cv(rf_pipe, X, y, n_splits=cv_splits, random_state=42, n_jobs=cv_n_jobs)
    print("\nRandom Forest (baseline) CV summary:")
    for k, v in rf_cv_summary.items():
        print(f"  {k}: {v:.4f}")

    save_cv_summary(rf_cv_summary, "RandomForest", FAST_MODE)
    # -------------------------
    # Bagging baseline
    # -------------------------
    base_tree = DecisionTreeRegressor(random_state=42)
    bag = BaggingRegressor(
        estimator=base_tree,
        n_estimators=bag_estimators,
        random_state=42,
        n_jobs=-1,
    )

    bag_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", bag),
        ]
    )

    bag_cv_summary, _ = evaluate_with_cv(bag_pipe, X, y, n_splits=cv_splits, random_state=42, n_jobs=cv_n_jobs)
    print("\nBagging (baseline) CV summary:")
    for k, v in bag_cv_summary.items():
        print(f"  {k}: {v:.4f}")

    save_cv_summary(bag_cv_summary, "Bagging", FAST_MODE)
    # -------------------------
    # Hyperparameter tuning za Random Forest (samo u final modu)
    # -------------------------
    best_rf_pipe = rf_pipe

    if do_grid_search:
        rf_tune = RandomForestRegressor(random_state=42, n_jobs=-1)

        rf_tune_pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", rf_tune),
            ]
        )

        # Razuman grid (možeš ga proširiti kasnije)
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
            scoring=rmse_scorer,  # negativni RMSE
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X, y)
        best_rmse = -grid.best_score_

        print("\nRandom Forest tuning:")
        print("  Best CV RMSE:", round(best_rmse, 4))
        print("  Best params:", grid.best_params_)

        best_rf_pipe = grid.best_estimator_

    # -------------------------
    # Permutation feature importance (na odabranom RF modelu)
    # -------------------------
    best_rf_pipe.fit(X, y)
    imp_df = compute_permutation_importance(best_rf_pipe, X, y, n_repeats=perm_repeats, random_state=42)
    imp_df["fast_mode"] = FAST_MODE
    imp_df.to_csv("rf_permutation_importance.csv", index=False)

    print("\nTop 15 permutation importances (R2 scoring):")
    print(imp_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
