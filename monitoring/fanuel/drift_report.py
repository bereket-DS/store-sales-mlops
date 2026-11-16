import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set paths relative to the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = REPO_ROOT / "data" / "processed" / "fanuel" / "valid_predictions.parquet"
OUT_DIR = REPO_ROOT / "monitoring" / "fanuel"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_drift_report():
    print("Loading validation predictions...")
    df = pd.read_parquet(PRED_PATH)

    # Calculate basic drift metrics
    actual_mean = df["sales"].mean()
    pred_mean = df["pred_sales"].mean()
    drift_ratio = pred_mean / actual_mean if actual_mean > 0 else 0

    print(f"Actual Mean: {actual_mean:.2f}")
    print(f"Predicted Mean: {pred_mean:.2f}")
    print(f"Drift Ratio (Pred/Actual): {drift_ratio:.4f}")

    # Generate a distribution plot
    plt.figure(figsize=(10, 5))
    plt.hist(np.log1p(df["sales"]), bins=50, alpha=0.5, label="Actual Sales (log1p)", color="blue")
    plt.hist(np.log1p(df["pred_sales"]), bins=50, alpha=0.5, label="Predicted Sales (log1p)", color="orange")
    plt.title("Actual vs Predicted Sales Distribution (Validation Set)")
    plt.xlabel("log1p(Sales)")
    plt.ylabel("Frequency")
    plt.legend()

    # Save plot
    plot_path = OUT_DIR / "distribution_drift.png"
    plt.savefig(plot_path)
    print(f"Saved distribution plot to {plot_path}")


if __name__ == "__main__":
    generate_drift_report()