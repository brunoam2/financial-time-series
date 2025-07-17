from pathlib import Path
from joblib import dump
import pandas as pd

from ..config import MODEL_PATH, TARGET_COLUMN, WINDOW_SIZE, HORIZON
from ..models.lstm_model import LSTMModel
from ..models.gru_model import GRUModel
from ..models.transformer_model import TransformerModel
from ..models.xgboost_model import XGBoostModel

def train_model(model_name: str, X_train, y_train=None, X_val=None, y_val=None, horizon: int = HORIZON) -> Path:
    """Entrena un modelo y guarda el resultado."""

    model_class_map = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "transformer": TransformerModel,
        "xgboost": XGBoostModel,
    }

    if model_name not in model_class_map:
        raise ValueError(f"Modelo '{model_name}' no está soportado")

    model_class = model_class_map[model_name]
    shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim > 2 else (X_train.shape[1],)
    model_instance = model_class(shape, horizon=horizon)
    model_instance.fit(X_train, y_train, validation_data=(X_val, y_val))

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_output_path = MODEL_PATH / f"{model_name}_{TARGET_COLUMN}_w{WINDOW_SIZE}_h{horizon}.pkl"
    dump(model_instance, model_output_path)
    print(f"Modelo '{model_name}' entrenado y guardado en: {model_output_path}")
    return model_output_path
