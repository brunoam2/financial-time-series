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

def create_sliding_windows(
    dataframe: pd.DataFrame,
    window_size: int,
    normalization_method: str = "standard",
    exclude_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Genera secuencias de entrada ``X`` y etiquetas ``y`` mediante una ventana
    deslizante. Cada ventana se normaliza de forma independiente para evitar
    *data leakage*.

    El argumento ``exclude_columns`` permite descartar columnas antes de crear
    las ventanas, por ejemplo para entrenar con solo variables exógenas."""
    if window_size <= 0 or len(dataframe) <= window_size:
        raise ValueError("El tamaño de ventana debe ser positivo y menor que el número de filas del DataFrame.")

    features_sequences = []
    target_values = []
    target_scaling_parameters = []

    exclude_columns = exclude_columns or []

    for start_index in range(len(dataframe) - window_size):
        end_index = start_index + window_size
        window_raw = dataframe.iloc[start_index:end_index]
        window_data = window_raw.drop(columns=exclude_columns, errors="ignore").copy()
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
            

        # Calcular parámetros de normalización del objetivo usando la ventana
        target_slice = window_raw[[TARGET_COLUMN]]
        _, target_params_dict = normalize_dataframe_columns(
            target_slice,
            method="minmax" if TARGET_SCALER == "minmax" else "standard",
        )
        target_parameters = target_params_dict[TARGET_COLUMN]
        target_normalized = normalize_value(target_value, target_parameters, TARGET_SCALER)

        features_sequences.append(normalized_window.values)
        target_values.append(target_normalized)
        target_scaling_parameters.append(target_parameters)

    return np.array(features_sequences), np.array(target_values), target_scaling_parameters
