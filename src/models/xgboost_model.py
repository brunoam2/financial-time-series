import numpy as np
from xgboost import XGBRegressor
from src.config import EARLY_STOPPING_PATIENCE


class XGBoostModel:
    """Modelo XGBoost entrenado sobre ventanas deslizantes."""

    def __init__(self, input_shape: tuple) -> None:
        self.window_size, self.n_features = input_shape
        self.model = XGBRegressor(objective="reg:squarederror")

    def _reshape(self, X):
        return X.reshape(X.shape[0], -1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data=None,
        **kwargs,
    ) -> None:
        X_train_reshaped = self._reshape(X_train)

        self.model.fit(
            X_train_reshaped,
            y_train,
            verbose=True,
            **kwargs,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._reshape(X))