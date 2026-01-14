import os
import pandas as pd
import matplotlib.pyplot as plt


METRICS_CSV = "results_cv_metric.csv"
IMPORTANCE_CSV = "rf_permutation_importance.csv"
OUT_DIR = "figures"


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_final_metrics():
    df = pd.read_csv(METRICS_CSV)

    # uzmi samo finalne runove (fast_mode == False)
    df = df[df["fast_mode"] == False].copy()

    if df.empty:
        raise ValueError("Nema redova s fast_mode == False u results_cv_metric.csv")

    # Ako ima duplikata (više pokretanja), uzmi zadnji zapis za svaki model
    df["_row"] = range(len(df))
    df = df.sort_values("_row").groupby("model", as_index=False).tail(1).drop(columns=["_row"])

    # Sortiraj stabilno: RandomForest pa Bagging (ili abecedno ako želiš)
    preferred_order = ["DecisionTree", "Bagging", "RandomForest"]
    df["model"] = pd.Categorical(df["model"], categories=preferred_order, ordered=True)
    df = df.sort_values("model")

    return df


def plot_metrics_bar(df_metrics: pd.DataFrame):
    # Bar chart za RMSE, MAE, R2 u tri odvojena grafa (čisto i pregledno)
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

        # upiši vrijednosti iznad stupaca
        for i, v in enumerate(df_metrics[metric].values):
            plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

        out_path = os.path.join(OUT_DIR, f"{metric}_bar.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")


def load_importances_top(n=10):
    df = pd.read_csv(IMPORTANCE_CSV)

    # uzmi samo finalni run (fast_mode == False)
    df = df[df["fast_mode"] == False].copy()

    if df.empty:
        raise ValueError("Nema redova s fast_mode == False u rf_permutation_importance.csv")

    # top N po importance_mean
    df = df.sort_values("importance_mean", ascending=False).head(n)
    return df


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
    # graf sa error barovima (stability)
    plt.figure(figsize=(10, 6))
    plt.barh(
        df_imp["feature"],
        df_imp["importance_mean"],
        xerr=df_imp["importance_std"]
    )
    plt.gca().invert_yaxis()
    plt.title("Permutation importance + varijabilnost (std)")
    plt.xlabel("Mean decrease in R² (± std)")
    plt.ylabel("Značajka (feature)")

    out_path = os.path.join(OUT_DIR, "rf_permutation_importance_top10_std.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ensure_out_dir()

    # 1) metrički grafovi
    df_metrics = load_final_metrics()
    plot_metrics_bar(df_metrics)

    # 2) importance grafovi
    df_imp_top10 = load_importances_top(n=10)
    plot_importances_bar(df_imp_top10)
    plot_importances_with_std(df_imp_top10)

    print("\nGotovo. Grafovi su u folderu:", OUT_DIR)


if __name__ == "__main__":
    main()