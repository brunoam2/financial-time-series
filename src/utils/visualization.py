import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_actual_vs_predicted(actual_values: pd.Series, predicted_values: pd.Series, title: str = "") -> None:
    # Grafica los valores reales frente a los predichos
    plt.figure(figsize=(10, 5))
    plt.plot(actual_values, label="Real", linewidth=2)
    plt.plot(predicted_values, label="Predicho", linestyle="--")
    plt.title(title or "Comparación: Real vs. Predicho")
    plt.xlabel("Índice temporal")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(residuals: pd.Series, title: str = "Residuos del modelo") -> None:
    # Grafica los residuos de una predicción
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, color="tab:red", linewidth=1)
    plt.title(title)
    plt.xlabel("Índice temporal")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_actual_vs_predicted(
    actual_values: pd.Series,
    predicted_values: pd.Series,
    output_path: Path | str,
    title: str = "",
) -> None:
    """Guarda la gráfica de valores reales vs predichos en ``output_path``."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_values, label="Real", linewidth=2)
    ax.plot(predicted_values, label="Predicho", linestyle="--")
    ax.set_title(title or "Comparación: Real vs. Predicho")
    ax.set_xlabel("Índice temporal")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "Comparativa de métricas",
) -> None:
    """Crea un gráfico de barras comparando métricas por modelo."""
    ax = metrics_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()