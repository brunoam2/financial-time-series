import pandas as pd
from prophet import Prophet


class ProphetModel:
    """Modelo Prophet para series temporales."""

    def __init__(self) -> None:
        self.model = Prophet()
        self.last_date = None

    def fit(self, X_train: pd.DataFrame, y_train=None) -> None:
        df = X_train.rename(columns={X_train.columns[0]: "ds", X_train.columns[1]: "y"}) if isinstance(X_train, pd.DataFrame) else X_train
        if "ds" not in df or "y" not in df:
            raise ValueError("Se requiere un DataFrame con columnas 'ds' y 'y'")
        self.last_date = df["ds"].iloc[-1]
        self.model.fit(df)

    def predict(self, X) -> pd.Series:
        steps = X if isinstance(X, int) else len(X)
        future_dates = pd.date_range(start=self.last_date + pd.Timedelta(days=1), periods=steps)
        future = pd.DataFrame({"ds": future_dates})
        forecast = self.model.predict(future)
        return forecast["yhat"]