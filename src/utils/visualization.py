import matplotlib.pyplot as plt
import pandas as pd


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