from pathlib import Path
from joblib import dump
import pandas as pd

from ..config import MODEL_PATH, NORMALIZATION_METHOD, TARGET_ASSET, WINDOW_SIZE
from ..models.lstm_model import LSTMModel

def train_model(model_name: str, X_train, y_train, X_val, y_val) -> Path:
    """Entrena un modelo y lo guarda en disco"""

    model_class_map = {
        "lstm": LSTMModel
    }

    if model_name not in model_class_map:
        raise ValueError(f"Modelo '{model_name}' no está soportado. Usa uno de: {list(model_class_map.keys())}")

    model_class = model_class_map[model_name]

    model = model_class((X_train.shape[1], X_train.shape[2]) if len(X_train.shape) > 2 else (X_train.shape[1],))
    model.fit(X_train, y_train, validation_data=(X_val, y_val))

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    file_name = f"{model_name}_{TARGET_ASSET}_w{WINDOW_SIZE}_{NORMALIZATION_METHOD}.pkl"
    model_file = MODEL_PATH / file_name
    dump(model, model_file)

    return model_file
