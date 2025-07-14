from pathlib import Path
from joblib import dump

from ..config import MODEL_PATH, NORMALIZATION_METHOD, TARGET_COLUMN, WINDOW_SIZE
from ..models.lstm_model import LSTMModel
from ..models.gru_model import GRUModel
from ..models.transformer_model import TransformerModel
from ..models.xgboost_model import XGBoostModel
from ..models.arima_model import ARIMAModel
from ..models.prophet_model import ProphetModel

def train_model(model_name: str, X_train, y_train, X_val=None, y_val=None) -> Path:
    """Entrena un modelo y guarda el resultado."""

    model_class_map = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "transformer": TransformerModel,
        "xgboost": XGBoostModel,
        "arima": ARIMAModel,
        "prophet": ProphetModel,
    }

    if model_name not in model_class_map:
        raise ValueError(f"Modelo '{model_name}' no está soportado")

    cls = model_class_map[model_name]
    if model_name in {"arima", "prophet"}:
        model = cls()
        model.fit(X_train, y_train)
    else:
        shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim > 2 else (X_train.shape[1],)
        model = cls(shape)
        model.fit(X_train, y_train, validation_data=(X_val, y_val))

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_file = MODEL_PATH / f"{model_name}_{TARGET_COLUMN}_w{WINDOW_SIZE}_{NORMALIZATION_METHOD}.pkl"
    dump(model, model_file)
    return model_file
