import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from src.utils.metrics import (
    calculate_mean_absolute_error,
    calculate_root_mean_squared_error,
    calculate_mean_absolute_percentage_error
)
from src.utils.data_loading import load_series
from src.utils.preprocessing import (
     normalize_values,
    create_sliding_windows,
)
from src.config import (
    MODEL_PATH,
    SELECTED_VARIABLES,
    TEST_SIZE,
    WINDOW_SIZE,
    NORMALIZATION_METHOD,
)


def evaluate_model(model_name: str, series_ticker: str) -> dict[str, float]:
    # Carga y prepara los datos
    df = load_series(tickers=[series_ticker], columns=SELECTED_VARIABLES)
    df_norm = normalize_values(df, NORMALIZATION_METHOD)
    X, y = create_sliding_windows(df_norm[SELECTED_VARIABLES], window_size=WINDOW_SIZE)

    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    # Carga el modelo entrenado
    model_path = MODEL_PATH / f"{model_name}_{series_ticker}.pkl"
    model = load(model_path)

    # Genera predicciones
    predictions = model.predict(X_test)

    # Evalúa las predicciones
    metrics = {
        "MAE": calculate_mean_absolute_error(y_test, predictions),
        "RMSE": calculate_root_mean_squared_error(y_test, predictions)
    }

    return metrics