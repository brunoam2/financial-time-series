import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config import TARGET_COLUMN, TARGET_SCALER


def normalize_dataframe_columns(dataframe: pd.DataFrame, method: str = "standard") -> tuple[pd.DataFrame, dict[str, tuple]]:
    if method == "standard":
        scaler_class = StandardScaler
    elif method == "minmax":
        scaler_class = lambda: MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Método de normalización no soportado: {method}")
    normalized_dataframe = dataframe.copy()
    columns_selected_for_normalization = dataframe.columns
    scaler_parameters = {}

    for column_name in columns_selected_for_normalization:
        scaler = scaler_class() if callable(scaler_class) else scaler_class
        normalized_dataframe[column_name] = scaler.fit_transform(dataframe[[column_name]])
        if method == "standard":
            scaler_parameters[column_name] = (scaler.mean_[0], scaler.scale_[0])
        else:
            scaler_parameters[column_name] = (scaler.data_min_[0], scaler.data_max_[0])

    return normalized_dataframe, scaler_parameters

def normalize_value(original_value: float, scaler_parameters: tuple, method: str) -> float:
    if method == "standard":
        mean_value, standard_deviation = scaler_parameters
        return (original_value - mean_value) / standard_deviation
    elif method == "minmax":
        minimum_value, maximum_value = scaler_parameters
        return (original_value - minimum_value) / (maximum_value - minimum_value)
    else:
        raise ValueError(f"Método de normalización no soportado: {method}")

def denormalize_value(normalized_value: float, scaler_parameters: tuple, method: str) -> float:
    if method == "standard":
        mean_value, standard_deviation = scaler_parameters
        return normalized_value * standard_deviation + mean_value
    elif method == "minmax":
        minimum_value, maximum_value = scaler_parameters
        return normalized_value * (maximum_value - minimum_value) + minimum_value
    else:
        raise ValueError(f"Método de normalización no soportado: {method}")

def create_sliding_windows(dataframe: pd.DataFrame, window_size: int, normalization_method: str = "standard") -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Genera secuencias de entrada (X) y etiquetas (y) a partir de un DataFrame utilizando una ventana deslizante,
    normalizando cada ventana por separado para evitar data leakage"""
    if window_size <= 0 or len(dataframe) <= window_size:
        raise ValueError("El tamaño de ventana debe ser positivo y menor que el número de filas del DataFrame.")

    features_sequences = []
    target_values = []
    target_scaling_parameters = []

    for start_index in range(len(dataframe) - window_size):
        end_index = start_index + window_size
        window_data = dataframe.iloc[start_index:end_index].copy()
        target_value = dataframe[TARGET_COLUMN].iloc[end_index]

        # Selección de columnas por tipo
        return_columns = [col for col in window_data.columns if "Return" in col]
        price_columns = [col for col in window_data.columns if col.endswith("_Close")]
        # Considerar volumen e indicadores técnicos como "otras" columnas
        technical_columns = [col for col in window_data.columns if col not in return_columns + price_columns]

        normalized_window = window_data.copy()
        normalization_params = {}

        if return_columns:
            normalized_returns, return_params = normalize_dataframe_columns(
                window_data[return_columns], method="standard"
            )
            normalized_window[return_columns] = normalized_returns
            normalization_params.update(return_params)

        if price_columns:
            normalized_prices, price_params = normalize_dataframe_columns(
                window_data[price_columns], method="minmax"
            )
            normalized_window[price_columns] = normalized_prices
            normalization_params.update(price_params)

        if technical_columns:
            normalized_tech, tech_params = normalize_dataframe_columns(
                window_data[technical_columns], method="standard"
            )
            normalized_window[technical_columns] = normalized_tech
            

        # Extraer y guardar solo los parámetros del target
        if TARGET_COLUMN not in normalization_params:
            raise ValueError(f"No se encontraron parámetros de normalización para {TARGET_COLUMN}.")

        target_parameters = normalization_params[TARGET_COLUMN]
        target_normalized = normalize_value(target_value, target_parameters, TARGET_SCALER)

        features_sequences.append(normalized_window.values)
        target_values.append(target_normalized)
        target_scaling_parameters.append(target_parameters)

    return np.array(features_sequences), np.array(target_values), target_scaling_parameters
