import sys
from pathlib import Path
from joblib import load
import random
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_PATH,
    SELECTED_FEATURES,
    WINDOW_SIZE,
    MODEL_TYPE,
    VALIDATION_START,
    VALIDATION_END,
    PREDICTIONS_PATH,
    SEED,
    TARGET_COLUMN,
)
from src.utils.preprocessing import fill_missing_values, create_sliding_windows, denormalize_value
from src.pipeline.train import train_model
from src.utils.metrics import calculate_all_metrics
from src.utils.visualization import plot_actual_vs_predicted, plot_residuals

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_PATH / "combined_data.csv", index_col=0, parse_dates=True)
    cols = [TARGET_COLUMN] + [c for c in SELECTED_FEATURES if c in df.columns]
    df = fill_missing_values(df[cols])

    X_all, y_all, params = create_sliding_windows(df, WINDOW_SIZE)
    dates = df.index[WINDOW_SIZE:]
    val_mask = (dates >= pd.to_datetime(VALIDATION_START)) & (dates <= pd.to_datetime(VALIDATION_END))

    X_train, y_train = X_all[~val_mask], y_all[~val_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    params_val = params[val_mask]

    model_type = MODEL_TYPE.lower()
    if model_type == "arima":
        series = df[TARGET_COLUMN].iloc[WINDOW_SIZE:][~val_mask]
        series.index.freq = pd.infer_freq(series.index)
        model_path = train_model("arima", series, None, None, None)
    elif model_type == "prophet":
        prophet_df = df[[TARGET_COLUMN]].reset_index().rename(columns={"index": "ds", TARGET_COLUMN: "y"})
        prophet_df = prophet_df.iloc[WINDOW_SIZE:][~val_mask]
        model_path = train_model("prophet", prophet_df, None, None, None)
    else:
        model_path = train_model(model_type, X_train, y_train, X_val, y_val)

    model = load(model_path)
    preds = model.predict(X_val)

    real = np.array([denormalize_value(v, p) for v, p in zip(y_val, params_val)])
    preds = np.array([denormalize_value(v, p) for v, p in zip(preds, params_val)])

    mae, rmse = calculate_all_metrics(real, preds)
    print(f"MAE: {mae}\nRMSE: {rmse}")

    plot_actual_vs_predicted(pd.Series(real, index=dates[val_mask]), pd.Series(preds, index=dates[val_mask]))
    plot_residuals(pd.Series(real - preds, index=dates[val_mask]))

    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)
    out_file = PREDICTIONS_PATH / f"{model_type}_{TARGET_COLUMN}.csv"
    pd.DataFrame({"date": dates[val_mask], "real": real, "predicted": preds}).to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
