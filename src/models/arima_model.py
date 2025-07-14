import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """Modelo ARIMA para series temporales univariadas."""

    def __init__(self, order=(1, 1, 1)) -> None:
        self.order = order
        self.model = None
        self.last_date = None

    def fit(self, X_train: pd.Series, y_train=None) -> None:
        # Use flattening that is safe for both Series and DataFrame
        if isinstance(X_train, pd.Series):
            series = X_train
        else:
            # For DataFrame or array-like, flatten to 1D Series
            series = pd.Series(X_train.values.ravel(), index=X_train.index)
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Se requiere un índice de fechas para ARIMA")
        if series.std() < 1e-8:
            raise ValueError("La serie temporal parece ser constante o tiene muy poca variabilidad.")
        self.last_date = series.index[-1]
        self.model = ARIMA(series, order=self.order).fit()

    def predict(self, X) -> pd.Series:
        steps = X if isinstance(X, int) else len(X)
        forecast = self.model.forecast(steps=steps)
        return forecast.astype(float)