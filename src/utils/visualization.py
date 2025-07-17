import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_actual_vs_predicted(actual_values: pd.Series | pd.DataFrame, predicted_values: pd.Series | pd.DataFrame, title: str = "") -> None:
    actual = pd.DataFrame(actual_values)
    predicted = pd.DataFrame(predicted_values)

    # Aplanar y resetear índice para graficar como series combinadas
    actual_flat = actual.values.flatten()
    predicted_flat = predicted.values.flatten()

    # Crear índice temporal unificado
    index = range(len(actual_flat))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(index, actual_flat, label="Real", linewidth=2)
    ax.plot(index, predicted_flat, label="Predicho", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Índice temporal")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
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
    actual_values: pd.Series | pd.DataFrame,
    predicted_values: pd.Series | pd.DataFrame,
    output_path: Path | str,
    title: str = "",
) -> None:
    actual = pd.DataFrame(actual_values)
    predicted = pd.DataFrame(predicted_values)

    actual_flat = actual.values.flatten()
    predicted_flat = predicted.values.flatten()

    index = range(len(actual_flat))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(index, actual_flat, label="Real", linewidth=2)
    ax.plot(index, predicted_flat, label="Predicho", linestyle="--")

    ax.set_title(title)
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


def plot_combined_diagnostics(
    actual_values: pd.Series | pd.DataFrame,
    predicted_values: pd.Series | pd.DataFrame,
    residuals: pd.Series,
    title: str = "",
) -> None:
    """Realiza gráficos combinados de valores reales vs predichos y residuos."""
    plot_actual_vs_predicted(actual_values, predicted_values, title=title)
    plot_residuals(residuals, title=f"Residuos - {title}" if title else "Residuos del modelo")


def plot_error_by_horizon_step(actual_values, predicted_values, metric_fn, title="Error por paso de horizonte"):
    actual = pd.DataFrame(actual_values)
    predicted = pd.DataFrame(predicted_values)
    errors = [metric_fn(actual.iloc[:, i], predicted.iloc[:, i]) for i in range(predicted.shape[1])]
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, len(errors) + 1), errors)
    plt.title(title)
    plt.xlabel("Paso")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()