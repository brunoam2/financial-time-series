from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.optimizers import Adam
from src.config import EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA


class CNNModel:
    """Modelo CNN 1D para series temporales multivariables con horizonte múltiple."""

    def __init__(self, input_shape: tuple, horizon: int = 1) -> None:
        self.horizon = horizon
        self.model = Sequential([
            Input(shape=input_shape),
            Conv1D(32, kernel_size=3, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(horizon),
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    def fit(
        self,
        X_train,
        y_train,
        epochs: int = 150,
        batch_size: int = 32,
        validation_data=None,
    ) -> None:
        """Entrena el modelo CNN."""
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
        )
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1,
        )

    def predict(self, X, horizon: int | None = None):
        horizon = horizon or self.horizon
        predictions = self.model.predict(X, verbose=1)
        if horizon == 1:
            return predictions.flatten()
        return predictions
