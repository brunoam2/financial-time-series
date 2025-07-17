import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COLUMN, WINDOW_SIZE, HORIZON


def normalize_dataframe_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple]]:
    normalized_dataframe = dataframe.copy()
    scaler_parameters = {}

    for column_name in dataframe.columns:
        scaler = StandardScaler()
        normalized_dataframe[column_name] = scaler.fit_transform(dataframe[[column_name]])
        scaler_parameters[column_name] = (scaler.mean_[0], scaler.scale_[0])

    return normalized_dataframe, scaler_parameters

def normalize_values(original_values: list[float], scaler_parameters: tuple) -> list[float]:
    mean_value, standard_deviation = scaler_parameters
    return [(v - mean_value) / standard_deviation for v in original_values]

def denormalize_values(normalized_values: list[float], scaler_parameters: tuple) -> list[float]:
    mean_value, standard_deviation = scaler_parameters
    return [v * standard_deviation + mean_value for v in normalized_values]

def create_sliding_windows(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    
    # Se valida que el tamaño de la ventana y el horizonte sean válidos
    if WINDOW_SIZE <= 0 or len(dataframe) <= WINDOW_SIZE + HORIZON:
        raise ValueError("El tamaño de ventana y el horizonte deben ser válidos para el número de filas del DataFrame.")

    # Se inicializan los conjuntos que almacenarán las ventanas
    X = []
    y = []
    target_scaling_parameters = []

    # Se establece el número máximo de ventanas que se pueden crear
    max_start = len(dataframe) - WINDOW_SIZE - HORIZON + 1

    for start_index in range(0, max_start):

        # Se toman los valores de la ventana y el objetivo
        end_index = start_index + WINDOW_SIZE
        window_data = dataframe.iloc[start_index:end_index]
        # target_data siempre es una lista, incluso si HORIZON == 1, para facilitar el procesamiento homogéneo
        target_data = dataframe[TARGET_COLUMN].iloc[end_index : end_index + HORIZON].tolist()

        # Se normalizan los datos de la ventana y se extraen los parámetros de escalado del objetivo
        normalized_window, normalization_params = normalize_dataframe_columns(window_data)
        target_parameters = normalization_params[TARGET_COLUMN]

        target_normalized = normalize_values(target_data, target_parameters)

        X.append(normalized_window.values)
        y.append(target_normalized)
        target_scaling_parameters.append(target_parameters)

    return np.array(X), np.array(y), np.array(target_scaling_parameters)
