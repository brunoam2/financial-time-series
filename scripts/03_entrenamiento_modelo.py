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
    DATA_PATH,
    WINDOW_SIZE,
    MODEL_TYPE,
    VALIDATION_START,
    VALIDATION_END,
    TARGET_SCALER,
    PREDICTIONS_PATH,
    SEED,
    TARGET_COLUMN,
)
from src.utils.preprocessing import denormalize_value
from src.pipeline.train import train_model
from src.utils.metrics import calculate_all_metrics
from src.utils.visualization import plot_actual_vs_predicted, plot_residuals

random.seed(SEED)
np.random.seed(SEED)

# Load processed data
X_train = np.load(PROCESSED_DATA_PATH / "X_train.npy")
X_val = np.load(PROCESSED_DATA_PATH / "X_val.npy")
y_train = np.load(PROCESSED_DATA_PATH / "y_train.npy")
y_val = np.load(PROCESSED_DATA_PATH / "y_val.npy")

with open(PROCESSED_DATA_PATH / "y_val_params.pkl", "rb") as f:
    params_val = pickle.load(f)

# Load dates for validation set
df = pd.read_csv( DATA_PATH / "combined_data.csv", index_col=0, parse_dates=True)
dates = df.index[WINDOW_SIZE:]
val_mask = (dates >= pd.to_datetime(VALIDATION_START)) & (dates <= pd.to_datetime(VALIDATION_END))

train_dates = (dates < pd.to_datetime(VALIDATION_START))

model_type = MODEL_TYPE.lower()
if model_type == "arima":
    # Crear una serie temporal univariante de entrenamiento basada en la target column
    series = df[TARGET_COLUMN].iloc[WINDOW_SIZE:][train_dates]
    series.index = pd.date_range(start=series.index[0], periods=len(series), freq='B')
    model_path = train_model("arima", series)
elif model_type == "prophet":
    prophet_df = df[[TARGET_COLUMN]].iloc[WINDOW_SIZE:][train_dates].copy()
    prophet_df = prophet_df.reset_index().rename(columns={"index": "ds", TARGET_COLUMN: "y"})
    model_path = train_model("prophet", prophet_df)
else:
    model_path = train_model(model_type, X_train, y_train, X_val, y_val)

model = load(model_path)
preds = model.predict(X_val)

real = np.array([denormalize_value(v, p, method=TARGET_SCALER) for v, p in zip(y_val, params_val)])
preds = np.array([denormalize_value(v, p, method=TARGET_SCALER) for v, p in zip(preds, params_val)])

mae, rmse = calculate_all_metrics(real, preds)
print(f"MAE: {mae}\nRMSE: {rmse}")

plot_actual_vs_predicted(pd.Series(real, index=dates[val_mask]), pd.Series(preds, index=dates[val_mask]))
plot_residuals(pd.Series(real - preds, index=dates[val_mask]))

PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)
out_file = PREDICTIONS_PATH / f"{model_type}_{TARGET_COLUMN}.csv"
pd.DataFrame({"date": dates[val_mask], "real": real, "predicted": preds}).to_csv(out_file, index=False)
