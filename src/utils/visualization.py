import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_actual_vs_predicted(actual_values: pd.Series | pd.DataFrame, predicted_values: pd.Series | pd.DataFrame, title: str = "") -> None:
    """Grafica valores reales vs predichos.

    Si ``predicted_values`` contiene múltiples horizontes (columnas), se genera
    una subgráfica por cada paso de predicción.
    """
    actual = pd.DataFrame(actual_values)
    predicted = pd.DataFrame(predicted_values)

    n_steps = predicted.shape[1]

    fig, axes = plt.subplots(n_steps, 1, figsize=(10, 4 * n_steps))
    if n_steps == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        step_col = predicted.columns[idx]
        ax.plot(actual.index, actual.iloc[:, idx if idx < actual.shape[1] else 0], label="Real", linewidth=2)
        ax.plot(predicted.index, predicted[step_col], label="Predicho", linestyle="--")
        step_title = f"Paso {idx + 1}" if n_steps > 1 else ""
        ax.set_title(f"{title} {step_title}".strip())
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
    """Guarda la gráfica de valores reales vs predichos en ``output_path``."""
    actual = pd.DataFrame(actual_values)
    predicted = pd.DataFrame(predicted_values)
    n_steps = predicted.shape[1]

    fig, axes = plt.subplots(n_steps, 1, figsize=(10, 4 * n_steps))
    if n_steps == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        step_col = predicted.columns[idx]
        ax.plot(actual.index, actual.iloc[:, idx if idx < actual.shape[1] else 0], label="Real", linewidth=2)
        ax.plot(predicted.index, predicted[step_col], label="Predicho", linestyle="--")
        step_title = f"Paso {idx + 1}" if n_steps > 1 else ""
        ax.set_title(f"{title} {step_title}".strip())
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