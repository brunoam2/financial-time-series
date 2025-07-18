import sys
from pathlib import Path
from joblib import load
import random
import numpy as np
import pandas as pd
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_PATH,
    HORIZON,
    MODEL_TYPE,
    PREDICTIONS_PATH,
    SEED,
    TARGET_COLUMN,
)

from src.pipeline.train import train_model
from src.utils.preprocessing import denormalize_values
from src.utils.metrics import calculate_all_metrics
from src.utils.visualization import plot_actual_vs_predicted, plot_residuals

random.seed(SEED)
np.random.seed(SEED)

# Load processed data
X = np.load(PROCESSED_DATA_PATH / "X.npy")
y = np.load(PROCESSED_DATA_PATH / "y.npy")
y_params = np.load(PROCESSED_DATA_PATH / "y_params.npy")

print(f"X.shape: {X.shape}, y.shape: {y.shape}, y_params.shape: {y_params.shape}")

# Verificar que el horizonte de y coincide con el configurado
real_horizon = y.shape[1]  # número de pasos de predicción por muestra

if real_horizon != HORIZON:
    raise ValueError(f"Los datos de y tienen un horizonte de {real_horizon}, pero se esperaba {HORIZON}.")

X_train = X[:-1]
y_train = y[:-1]
X_test = X[-1:]
y_test = y[-1:]
params_test = y_params[-1:]

model_type = MODEL_TYPE.lower()

model_path = train_model(model_type, X_train, y_train, X_test, y_test, horizon=HORIZON)
model = load(model_path)
preds = model.predict(X_test, horizon=HORIZON)

real = np.array([denormalize_values(y, p) for y, p in zip(y_test, params_test)])
preds = np.array([denormalize_values(y, p) for y, p in zip(preds, params_test)])

print(f"Adjusted real.shape: {real.shape}, preds.shape: {preds.shape}")

print(f"real.shape: {real.shape}")
print(f"preds.shape: {preds.shape}")
print(f"real: {real}")
print(f"preds: {preds}")

real_flat = np.concatenate(real)
preds_flat = np.concatenate(preds)
mae, rmse, mape, r2 = calculate_all_metrics(real_flat, preds_flat)
print(f"MAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}\nR²: {r2}")

real_df = pd.DataFrame(real, index=None)
pred_df = pd.DataFrame(preds, index=None)
plot_actual_vs_predicted(real_df, pred_df)
plot_residuals(pd.Series((real_df - pred_df).values.flatten(), index=None))

# Guardar predicciones como columnas 'real' y 'pred' con pasos como filas
comparison_df = pd.DataFrame({
    'real': real_flat,
    'pred': preds_flat
})
out_file = PREDICTIONS_PATH / f"{model_type}_{TARGET_COLUMN}_w{X.shape[1]}_h{HORIZON}.csv"
comparison_df.to_csv(out_file, index=False)
