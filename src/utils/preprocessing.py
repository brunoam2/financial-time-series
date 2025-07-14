import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config import NORMALIZATION_METHOD, TARGET_ASSET


def fill_missing_values(dataframe: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    # Rellena valores faltantes usando el método especificado
    if method == "ffill":
        return dataframe.ffill()
    elif method == "bfill":
        return dataframe.bfill()
    elif method == "drop":
        return dataframe.dropna()
    else:
        raise ValueError(f"Método de imputación no soportado: {method}")


def normalize_values(dataframe: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    scaler_cls = StandardScaler if method == "standard" else MinMaxScaler
    normalized = pd.DataFrame(index=dataframe.index)
    for column in dataframe.columns:
        scaler = scaler_cls()
        normalized[column] = scaler.fit_transform(dataframe[[column]])
    return normalized

def create_sliding_windows(df: pd.DataFrame, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera secuencias y etiquetas usando una ventana deslizante normalizada por ventana.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        X con forma (n_samples, window_size, n_features),
        y con forma (n_samples,),
        y_params con forma (n_samples, 2) conteniendo (min, max) o (mean, std) por muestra.
    """
    if window_size <= 0 or len(df) <= window_size:
        raise ValueError("window_size inválido para el DataFrame")

    target_column = f"{TARGET_ASSET}_Close"

    X, y, y_params = [], [], []
    for start in range(len(df) - window_size):
        end = start + window_size
        window = df.iloc[start:end].copy()
        window_norm = normalize_values(window, method=NORMALIZATION_METHOD)
        X.append(window_norm.values)

        target_seq = window[target_column].values
        y_value = df[target_column].iloc[end]

        if NORMALIZATION_METHOD == "standard":
            mean, std = target_seq.mean(), target_seq.std()
            y_params.append((mean, std))
            y.append((y_value - mean) / std if std else 0.0)

        else:
            min_val, max_val = target_seq.min(), target_seq.max()
            rng = max_val - min_val
            y_params.append((min_val, max_val))
            y.append((y_value - min_val) / rng if rng else 0.0)

    return np.array(X), np.array(y), np.array(y_params)


# Función para desnormalizar un valor dado los parámetros de normalización
def denormalize_value(y_normalized: float, param: tuple) -> float:
    if NORMALIZATION_METHOD == "standard":
        mean, std = param
        return y_normalized * std + mean
    elif NORMALIZATION_METHOD == "minmax":
        min_val, max_val = param
        return y_normalized * (max_val - min_val) + min_val
    else:
        raise ValueError(f"Método de normalización no soportado: {NORMALIZATION_METHOD}")
