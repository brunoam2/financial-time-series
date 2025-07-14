from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from src.config import EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA


class LSTMModel:
    """Modelo LSTM sencillo para series temporales."""

    def __init__(self, input_shape: tuple) -> None:
        self.model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(50),
                Dense(1)
            ]
        )
        self.model.compile(optimizer=Adam(), loss="mse")

    def fit(
        self,
        X_train,
        y_train,
        epochs: int = 150,
        batch_size: int = 32,
        validation_data=None,
    ) -> None:
        """Entrena el modelo LSTM."""

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

    def predict(self, X):
        """Genera predicciones utilizando el modelo entrenado."""
        predictions = self.model.predict(X, verbose=1)
        return predictions.flatten()