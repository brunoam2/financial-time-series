import sys
from pathlib import Path
import pandas as pd

# Añadir la raíz del proyecto al path para importar módulos
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.config import PREDICTIONS_PATH, FIGURES_PATH, METRICS_PATH
from src.utils.metrics import (
    calculate_mean_absolute_error,
    calculate_root_mean_squared_error,
    calculate_mean_absolute_percentage_error,
)
from src.utils.visualization import save_actual_vs_predicted, plot_metrics_comparison


def _load_series(csv_path: Path) -> pd.Series:
    """Carga un archivo CSV y devuelve la serie (index=datetime)."""
    df = pd.read_csv(csv_path, index_col=0)
    series = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    series.index = pd.to_datetime(series.index)
    return series


def main() -> None:
    """Compara todas las predicciones con los datos reales y genera métricas."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    real_files = list(PREDICTIONS_PATH.glob("real_*.csv"))
    if not real_files:
        print(f"No se encontró archivo de datos reales en {PREDICTIONS_PATH}")
        return

    real_path = real_files[0]
    data_name = real_path.stem.replace("real_", "")
    real_series = _load_series(real_path)

    metrics = []

    for pred_file in PREDICTIONS_PATH.glob(f"*_{data_name}.csv"):
        if pred_file.name.startswith("real_"):
            continue
        model_name = pred_file.stem.replace(f"_{data_name}", "")
        if not pred_file.exists():
            print(f"Advertencia: no existe {pred_file}")
            continue
        pred_series = _load_series(pred_file)

        aligned_real, aligned_pred = real_series.align(pred_series, join="inner")

        mae = calculate_mean_absolute_error(aligned_real.values, aligned_pred.values)
        rmse = calculate_root_mean_squared_error(aligned_real.values, aligned_pred.values)
        mape = calculate_mean_absolute_percentage_error(aligned_real.values, aligned_pred.values)

        metrics.append({"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape})

        fig_path = FIGURES_PATH / f"{model_name}_vs_real_{data_name}.png"
        save_actual_vs_predicted(aligned_real, aligned_pred, fig_path, title=f"{model_name} vs Real")

    if not metrics:
        print("No se encontraron predicciones para evaluar.")
        return

    metrics_df = pd.DataFrame(metrics).set_index("Model")
    metrics_file = METRICS_PATH / f"comparativa_modelos_{data_name}.csv"
    metrics_df.to_csv(metrics_file)

    metrics_fig = FIGURES_PATH / f"comparativa_metricas_{data_name}.png"
    plot_metrics_comparison(metrics_df, metrics_fig, title=f"Comparativa de métricas ({data_name})")


if __name__ == "__main__":
    main()
