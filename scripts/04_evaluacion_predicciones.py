import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    FIGURES_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    WINDOW_SIZE,
    HORIZON,
)
from src.utils.metrics import calculate_all_metrics


def load_prediction_files(window: int, horizon: int) -> list[Path]:
    """Return prediction CSV files matching the given window and horizon."""
    pattern = f"*w{window}_h{horizon}.csv"
    return sorted(PREDICTIONS_PATH.glob(pattern))


def main() -> None:
    """Compare models using their prediction files and save metrics and plots."""
    parser = argparse.ArgumentParser(description="Evaluación de predicciones")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE, help="Tamaño de la ventana")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Horizonte de predicción")
    args = parser.parse_args()

    window = args.window
    horizon = args.horizon

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)

    files = load_prediction_files(window, horizon)
    if not files:
        print(f"No se encontraron predicciones para w{window} h{horizon}.")
        return

    metrics: list[dict] = []
    predictions: list[tuple[str, pd.Series]] = []
    real_series = None

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            warnings.warn(f"No se pudo leer {fpath.name}: {exc}")
            continue

        if {"real", "pred"}.issubset(df.columns) and not df.empty:
            real = df["real"].values
            pred = df["pred"].values
        else:
            warnings.warn(f"{fpath.name} no tiene las columnas esperadas.")
            continue

        if real_series is None:
            real_series = real
        elif len(real_series) != len(real):
            warnings.warn(f"Longitud diferente en {fpath.name}; se ignora.")
            continue

        model_name = fpath.stem.split("_")[0]
        mae, rmse, mape, r2 = calculate_all_metrics(real, pred)
        metrics.append({"model": model_name, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2})
        predictions.append((model_name, pred))

        # Generar visualizaciones de errores individuales
        error = real - pred
        abs_error = abs(error)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].plot(error)
        axs[0].axhline(0, color="red", linestyle="--")
        axs[0].set_title(f"Residuos - {model_name}")
        axs[0].set_xlabel("Paso")
        axs[0].set_ylabel("Error")

        axs[1].hist(abs_error, bins=20, edgecolor="black")
        axs[1].set_title(f"Histograma Error Absoluto - {model_name}")
        axs[1].set_xlabel("Error")
        axs[1].set_ylabel("Frecuencia")

        axs[2].scatter(real, pred, alpha=0.6)
        axs[2].plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
        axs[2].set_title(f"Real vs Predicción - {model_name}")
        axs[2].set_xlabel("Real")
        axs[2].set_ylabel("Predicción")

        fig.suptitle(f"Análisis de Errores - {model_name} (w{window} h{horizon})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_error_path = FIGURES_PATH / f"errores_{model_name}_w{window}_h{horizon}.png"
        plt.savefig(fig_error_path)
        plt.close()

    if not metrics or real_series is None:
        print("No hay predicciones válidas para evaluar.")
        return

    metrics_df = pd.DataFrame(metrics).set_index("model")
    metrics_file = METRICS_PATH / f"comparativa_modelos_w{window}_h{horizon}.csv"
    metrics_df.to_csv(metrics_file)

    x = range(len(real_series))
    plt.figure(figsize=(10, 6))
    plt.plot(x, real_series, label="Real", linewidth=2)
    for name, pred in predictions:
        plt.plot(x, pred, label=name, linestyle="--")
    plt.xlabel("Paso")
    plt.ylabel("Valor")
    plt.title(f"Comparativa modelos (w{window} h{horizon})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = FIGURES_PATH / f"comparativa_modelos_w{window}_h{horizon}.png"
    plt.savefig(fig_path)
    plt.close()

    print(f"Métricas guardadas en {metrics_file}")
    print(f"Figura guardada en {fig_path}")


if __name__ == "__main__":
    main()

