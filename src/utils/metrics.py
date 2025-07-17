"""Funciones de métricas para evaluar predicciones."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mean_absolute_error(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Devuelve el MAE entre valores reales y predichos.
    Las entradas serán aplanadas para manejar predicciones multi-horizonte.
    """
    true_values = np.asarray(true_values).reshape(-1)
    predicted_values = np.asarray(predicted_values).reshape(-1)
    return mean_absolute_error(true_values, predicted_values)

def calculate_root_mean_squared_error(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Devuelve el RMSE entre valores reales y predichos.
    Las entradas serán aplanadas para manejar predicciones multi-horizonte.
    """
    true_values = np.asarray(true_values).reshape(-1)
    predicted_values = np.asarray(predicted_values).reshape(-1)
    return np.sqrt(mean_squared_error(true_values, predicted_values))

def calculate_mean_absolute_percentage_error(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Devuelve el MAPE entre valores reales y predichos.
    Las entradas serán aplanadas para manejar predicciones multi-horizonte.
    """
    true_values = np.asarray(true_values, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted_values, dtype=float).reshape(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs((true_values - predicted_values) / np.clip(np.abs(true_values), 1e-8, None))
    return np.mean(percentage_errors) * 100

def calculate_r2_score(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Devuelve el coeficiente de determinación R² entre valores reales y predichos.
    Las entradas serán aplanadas para manejar predicciones multi-horizonte.
    """
    true_values = np.asarray(true_values).reshape(-1)
    predicted_values = np.asarray(predicted_values).reshape(-1)
    return r2_score(true_values, predicted_values)


# Nueva función para calcular todas las métricas a la vez
def calculate_all_metrics(true_values: np.ndarray, predicted_values: np.ndarray) -> tuple[float, float, float]:
    """Calcula MAE, RMSE y R² en una sola llamada."""
    mae = calculate_mean_absolute_error(true_values, predicted_values)
    rmse = calculate_root_mean_squared_error(true_values, predicted_values)
    mape = calculate_mean_absolute_percentage_error(true_values, predicted_values)
    r2 = calculate_r2_score(true_values, predicted_values)
    return mae, rmse, mape, r2

