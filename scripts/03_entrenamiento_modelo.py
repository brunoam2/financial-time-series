import sys
from pathlib import Path
from joblib import load


import numpy as np
import random
import tensorflow as tf

import pandas as pd

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.config import SEED

# Importar función de visualización
from src.utils.visualization import plot_actual_vs_predicted, plot_residuals

# Establecer la semilla para reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src.config import (
    PROCESSED_DATA_PATH,
    TARGET_ASSET,
    SELECTED_FEATURES,
    WINDOW_SIZE,
    MODEL_TYPE,
    NORMALIZATION_METHOD,
    VALIDATION_START,
    VALIDATION_END,
    MODEL_PATH,
)
from src.utils.preprocessing import fill_missing_values, create_sliding_windows, denormalize_value
from src.pipeline.train import train_model
from src.utils.metrics import calculate_all_metrics

data_file = PROCESSED_DATA_PATH / "combined_data.csv"
df = pd.read_csv(data_file, index_col=0, parse_dates=True)

target_column = f"{TARGET_ASSET}_Close"
selected_columns = [target_column] + [col for col in SELECTED_FEATURES if col in df.columns]
df = df[selected_columns]

df = fill_missing_values(df)

X_all, y_all, y_params = create_sliding_windows(df, window_size=WINDOW_SIZE)
dates = df.index[WINDOW_SIZE:]

val_mask = (dates >= pd.to_datetime(VALIDATION_START)) & (dates <= pd.to_datetime(VALIDATION_END))
train_mask = dates < pd.to_datetime(VALIDATION_START)

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]
y_params_val = [y_params[i] for i, valid in enumerate(val_mask) if valid]

model_path = train_model(model_name=MODEL_TYPE.lower(), X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

model = load(model_path)
predictions = model.predict(X_val)

 
y_val_denorm = np.array([denormalize_value(y, p) for y, p in zip(y_val, y_params_val)])
predictions_denorm = np.array([denormalize_value(y, p) for y, p in zip(predictions, y_params_val)])

mae, rmse, mape = calculate_all_metrics(y_val_denorm, predictions_denorm)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE (%): {mape}")

# Visualización de predicciones vs realidad (con datos desnormalizados)
plot_actual_vs_predicted(pd.Series(y_val_denorm), pd.Series(predictions_denorm), title="Predicción vs Realidad (con datos desnormalizados)")
residuals = pd.Series(y_val_denorm) - pd.Series(predictions_denorm)
plot_residuals(residuals)
