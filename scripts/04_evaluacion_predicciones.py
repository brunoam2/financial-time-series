import sys
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load

# Ajustar ruta para poder importar paquetes del proyecto
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.config import (
    PROCESSED_DATA_PATH,
    TARGET_ASSET,
    SELECTED_FEATURES,
    WINDOW_SIZE,
    MODEL_TYPE,
    NORMALIZATION_METHOD,
    TEST_START,
    TEST_END,
    MODEL_PATH,
)
from src.utils.preprocessing import (
    fill_missing_values,
    create_sliding_windows,
    denormalize_value,
)
from src.utils.metrics import calculate_all_metrics
from src.utils.visualization import plot_actual_vs_predicted, plot_residuals


def main() -> None:
    # Cargar datos procesados
    data_file = PROCESSED_DATA_PATH / "combined_data.csv"
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Seleccionar columnas de interés
    target_column = f"{TARGET_ASSET}_Close"
    selected_columns = [target_column] + [col for col in SELECTED_FEATURES if col in df.columns]
    df = df[selected_columns]

    # Rellenar valores faltantes
    df = fill_missing_values(df)

    # Crear ventanas deslizantes normalizadas
    X_all, y_all, y_params = create_sliding_windows(df, window_size=WINDOW_SIZE)
    dates = df.index[WINDOW_SIZE:]

    # Filtrar rango de validación
    mask = (dates >= pd.to_datetime(TEST_START)) & (dates <= pd.to_datetime(TEST_END))
    X_test, y_test = X_all[mask], y_all[mask]
    y_params_test = [y_params[i] for i, valid in enumerate(mask) if valid]
    test_dates = dates[mask]

    # Cargar modelo entrenado
    model_file = MODEL_PATH / f"{MODEL_TYPE.lower()}_{TARGET_ASSET}_w{WINDOW_SIZE}_{NORMALIZATION_METHOD}.pkl"
    model = load(model_file)

    # Obtener predicciones
    predictions = model.predict(X_test)

    # Desnormalizar resultados
    y_test_denorm = np.array([denormalize_value(y, p) for y, p in zip(y_test, y_params_test)])
    predictions_denorm = np.array([denormalize_value(y, p) for y, p in zip(predictions, y_params_test)])

    # Calcular métricas
    mae, rmse, mape = calculate_all_metrics(y_test_denorm, predictions_denorm)
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE (%): {mape}")

    # Visualizaciones
    actual_series = pd.Series(y_test_denorm, index=test_dates)
    pred_series = pd.Series(predictions_denorm, index=test_dates)
    plot_actual_vs_predicted(actual_series, pred_series)
    residuals = actual_series - pred_series
    plot_residuals(residuals)


if __name__ == "__main__":
    main()