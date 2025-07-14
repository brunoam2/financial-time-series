from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense, MultiHeadAttention, GlobalAveragePooling1D
from keras.optimizers import Adam
from src.config import EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA


class TransformerModel:
    """Implementación simple de un Transformer para series temporales."""

    def __init__(self, input_shape: tuple) -> None:
        inputs = Input(shape=input_shape)
        x = MultiHeadAttention(num_heads=2, key_dim=input_shape[-1])(inputs, inputs)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(), loss="mse")

    def fit(
        self,
        X_train,
        y_train,
        epochs: int = 150,
        batch_size: int = 32,
        validation_data=None,
    ) -> None:
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
        predictions = self.model.predict(X, verbose=1)
        return predictions.flatten()