import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import PREDICTIONS_PATH, FIGURES_PATH, METRICS_PATH
from src.utils.metrics import (
    calculate_mean_absolute_error,
    calculate_root_mean_squared_error,
    calculate_mean_absolute_percentage_error,
)
from src.utils.visualization import save_actual_vs_predicted, plot_metrics_comparison


def main() -> None:
    """Compara todas las predicciones con los datos reales y genera métricas."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    # Cargar todos los archivos de predicciones que tengan la estructura esperada
    prediction_files = list(PREDICTIONS_PATH.glob("*.csv"))
    if not prediction_files:
        print(f"No se encontraron archivos de predicción en {PREDICTIONS_PATH}")
        return

    metrics = []

    for pred_file in prediction_files:
        df = pd.read_csv(pred_file)
        model_name = pred_file.stem.split("_")[0]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        if "real" in df.columns and "predicted" in df.columns:
            real_df = df[["real"]]
            pred_df = df[["predicted"]]
        else:
            real_cols = [c for c in df.columns if c.startswith("real_")]
            pred_cols = [c for c in df.columns if c.startswith("pred_")]
            if not real_cols or not pred_cols:
                print(f"Omitiendo {pred_file.name} por falta de columnas necesarias.")
                continue
            real_df = df[real_cols]
            pred_df = df[pred_cols]

        mae = calculate_mean_absolute_error(real_df.values, pred_df.values)
        rmse = calculate_root_mean_squared_error(real_df.values, pred_df.values)
        mape = calculate_mean_absolute_percentage_error(real_df.values, pred_df.values)

        metrics.append({"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape})

        fig_path = FIGURES_PATH / f"{model_name}_vs_real.png"
        save_actual_vs_predicted(real_df, pred_df, fig_path, title=f"{model_name} vs Real")

    if not metrics:
        print("No se encontraron predicciones válidas para evaluar.")
        return

    metrics_df = pd.DataFrame(metrics).set_index("Model")
    metrics_file = METRICS_PATH / "comparativa_modelos.csv"
    metrics_df.to_csv(metrics_file)

    metrics_fig = FIGURES_PATH / "comparativa_metricas.png"
    plot_metrics_comparison(metrics_df, metrics_fig, title="Comparativa de métricas")


if __name__ == "__main__":
    main()
