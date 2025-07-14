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
    SELECTED_FEATURES,
    WINDOW_SIZE,
    MODEL_TYPE,
    NORMALIZATION_METHOD,
    VALIDATION_START,
    VALIDATION_END,
    MODEL_PATH,
    SEED,
    TARGET_COLUMN,
)
from src.utils.preprocessing import fill_missing_values, create_sliding_windows, denormalize_value
from src.pipeline.train import train_model
from src.utils.metrics import calculate_all_metrics

data_file = PROCESSED_DATA_PATH / "combined_data.csv"
df = pd.read_csv(data_file, index_col=0, parse_dates=True)

print(df.head())
print(df.describe())

from src.config import TARGET_COLUMN
target_column = TARGET_COLUMN

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

if MODEL_TYPE.lower() == "arima":
    X_train_series = df[target_column].copy()
    X_train_series = X_train_series.iloc[WINDOW_SIZE:]
    X_train_series = X_train_series[train_mask].copy()
    X_train_series.index.freq = pd.infer_freq(X_train_series.index)
    print(X_train_series.head())
    model_path = train_model(
        model_name="arima",
        X_train=X_train_series,
        y_train=None,
        X_val=None,
        y_val=None
    )

elif MODEL_TYPE.lower() == "prophet":
    prophet_df = df[[target_column]].copy()
    prophet_df = prophet_df.reset_index().rename(columns={"index": "ds", target_column: "y"})
    prophet_df = prophet_df.iloc[WINDOW_SIZE:]
    prophet_df = prophet_df[train_mask].copy()
    print(prophet_df.head())
    model_path = train_model(
        model_name="prophet",
        X_train=prophet_df,
        y_train=None,
        X_val=None,
        y_val=None
    )

else:
    model_path = train_model(
        model_name=MODEL_TYPE.lower(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

model = load(model_path)

# Mostrar resumen del modelo ARIMA si aplica
if MODEL_TYPE.lower() == "arima":
    print(model.model.summary())
    import matplotlib.pyplot as plt
    residuals = model.model.resid
    plt.figure(figsize=(10, 4))
    plt.plot(residuals)
    plt.title("Residuos del modelo ARIMA")
    plt.tight_layout()
    plt.show()

print(type(X_val))
print(X_val[:5])

predictions = model.predict(X_val)

y_val = np.array(y_val)
predictions = np.array(predictions)

if len(y_val) != len(predictions):
    raise ValueError(f"Length mismatch: y_val has length {len(y_val)} but predictions has length {len(predictions)}")

y_val_denorm = np.array([denormalize_value(y, p) for y, p in zip(y_val, y_params_val)])
predictions_denorm = np.array([denormalize_value(y, p) for y, p in zip(predictions, y_params_val)])

print("y_val_denorm stats:", pd.Series(y_val_denorm).describe())
print("predictions_denorm stats:", pd.Series(predictions_denorm).describe())

mae, rmse = calculate_all_metrics(y_val_denorm, predictions_denorm)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Visualización de predicciones vs realidad (con datos desnormalizados)
plot_actual_vs_predicted(pd.Series(y_val_denorm), pd.Series(predictions_denorm), title="Predicción vs Realidad (con datos desnormalizados)")
residuals = pd.Series(y_val_denorm) - pd.Series(predictions_denorm)
plot_residuals(residuals)

import os

predictions_dir = project_root / "results" / "predictions"
predictions_dir.mkdir(parents=True, exist_ok=True)

df_preds = pd.DataFrame({
    "date": dates[val_mask],
    "real": y_val_denorm,
    "predicted": predictions_denorm
})
model_output_name = MODEL_TYPE.lower()
output_file = predictions_dir / f"{model_output_name}_{TARGET_COLUMN}.csv"
df_preds.to_csv(output_file, index=False)
print(f"Guardadas predicciones y valores reales en {output_file}")
