import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from src.config import EARLY_STOPPING_PATIENCE


class XGBoostModel:
    """Modelo XGBoost entrenado sobre ventanas deslizantes con soporte multi-horizonte."""

    def __init__(self, input_shape: tuple, horizon: int = 1) -> None:
        self.window_size, self.n_features = input_shape
        self.horizon = horizon
        base_model = XGBRegressor(objective="reg:squarederror")
        if horizon == 1:
            self.model = base_model
        else:
            self.model = MultiOutputRegressor(base_model)

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

    def predict(self, X: np.ndarray, horizon: int | None = None) -> np.ndarray:
        horizon = horizon or self.horizon
        preds = self.model.predict(self._reshape(X))
        if horizon == 1:
            return preds.flatten()
        return preds