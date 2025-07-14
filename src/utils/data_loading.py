import yfinance as yf
import pandas as pd
from src.config import DATA_PATH

def download_price_data(tickers_list, start_date, end_date, interval="1d", auto_adjust=True, excluded_tickers: list[str] = None):
   
    excluded_tickers = excluded_tickers or []
    tickers_with_volume = [t for t in tickers_list if t not in excluded_tickers]
    tickers_without_volume = excluded_tickers

    dataframe = yf.download(tickers_list, start=start_date, end=end_date, interval=interval, auto_adjust=auto_adjust, progress=False)

    if dataframe.empty:
        raise ValueError(f"No se encontraron datos para {tickers_list}.")

    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.swaplevel()
        dataframe.sort_index(axis=1, level=0, inplace=True)
        cols = []
        for ticker in tickers_with_volume:
            for attribute in ["Close", "Volume"]:
                if (ticker, attribute) in dataframe.columns:
                    cols.append((ticker, attribute))
        for ticker in tickers_without_volume:
            if (ticker, "Close") in dataframe.columns:
                cols.append((ticker, "Close"))

        dataframe = dataframe[cols]
        dataframe.columns = [f"{ticker}_{attribute.replace(' ', '')}" for ticker, attribute in dataframe.columns]

        dataframe.dropna(axis=0, inplace=True)
    else:
        raise ValueError("Se esperaba un MultiIndex en las columnas descargadas.")

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(DATA_PATH / "raw_price_data.csv")

    return dataframe