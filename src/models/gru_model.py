from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Input, GRU, Dense, Dropout
from keras.optimizers import Adam
from src.config import EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA


class GRUModel:
    """Modelo GRU sencillo para series temporales con soporte multi-horizonte."""

    def __init__(self, input_shape: tuple, horizon: int = 1) -> None:
        self.horizon = horizon
        self.model = Sequential([
            Input(shape=input_shape),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
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
        """Entrena el modelo GRU."""

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
        """Genera predicciones utilizando el modelo entrenado."""
        horizon = horizon or self.horizon
        predictions = self.model.predict(X, verbose=1)
        if horizon == 1:
            return predictions.flatten()
        return predictions
