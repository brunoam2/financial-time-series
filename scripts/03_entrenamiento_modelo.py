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
    HORIZON,
    MODEL_TYPE,
    TRAIN_END,
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

# Ejemplo de uso con horizonte de varios días:
# establece ``HORIZON = 7`` en ``src/config.py`` para entrenar un modelo que
# prediga los próximos 7 días.
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
if HORIZON == 1:
    dates = df.index[WINDOW_SIZE:]
else:
    dates = df.index[WINDOW_SIZE + HORIZON:]
val_mask = (dates >= pd.to_datetime(VALIDATION_START)) & (dates <= pd.to_datetime(VALIDATION_END))

train_dates = (dates < pd.to_datetime(VALIDATION_START))
validation_dates = (dates >= pd.to_datetime(VALIDATION_START)) & (dates <= pd.to_datetime(VALIDATION_END))

model_type = MODEL_TYPE.lower()

model_path = train_model(model_type, X_train, y_train, X_val, y_val, horizon=HORIZON)
model = load(model_path)
preds = model.predict(X_val, horizon=HORIZON)

def _denorm(array, params_list):
    output = []
    for arr, p in zip(array, params_list):
        if np.ndim(arr) == 0:
            output.append(denormalize_value(arr.item(), p, method=TARGET_SCALER))
        else:
            output.append([denormalize_value(v, p, method=TARGET_SCALER) for v in arr])
    return np.array(output)

real = _denorm(y_val, params_val)
preds = _denorm(preds, params_val)

mae, rmse = calculate_all_metrics(real, preds)
print(f"MAE: {mae}\nRMSE: {rmse}")

real_df = pd.DataFrame(real, index=dates[val_mask])
pred_df = pd.DataFrame(preds, index=dates[val_mask])
plot_actual_vs_predicted(real_df, pred_df)
plot_residuals(pd.Series((real_df - pred_df).values.flatten(), index=dates[val_mask].repeat(pred_df.shape[1])))

PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)
out_file = PREDICTIONS_PATH / f"{model_type}_{TARGET_COLUMN}.csv"
pd.concat(
    [real_df.add_prefix("real_"), pred_df.add_prefix("pred_")], axis=1
).assign(date=dates[val_mask]).to_csv(out_file, index=False)
