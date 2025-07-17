import yfinance as yf
import pandas as pd
from src.config import DATA_PATH

def download_price_data(tickers_list, start_date, end_date, interval="1d", auto_adjust=True):
   
    # Se descargan los datos desde yf
    dataframe = yf.download(tickers_list, start=start_date, end=end_date, interval=interval, auto_adjust=auto_adjust, progress=False)

    # Se comprueba que el df no está vacío
    if dataframe.empty:
        raise ValueError(f"No se encontraron datos para {tickers_list}.")

    #Se maneja el MultiIndex de las columnas
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.swaplevel()
        dataframe.sort_index(axis=1, level=0, inplace=True)
    else:
        raise ValueError("Se esperaba un MultiIndex en las columnas descargadas.")
    
    # Se seleccionan las columnas de cierre y volumen de cada ticker
    cols = []
    for ticker in tickers_list:
        for attribute in ["Close", "Volume"]:
            if (ticker, attribute) in dataframe.columns:
                cols.append((ticker, attribute))
    dataframe = dataframe[cols]
    dataframe.columns = [f"{ticker}_{attribute.replace(' ', '')}" for ticker, attribute in dataframe.columns]

    # Se eliminan filas con valores NaN
    dataframe.dropna(axis=0, inplace=True)

    # Se guarda el dataframe en un archivo CSV
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(DATA_PATH / "raw_price_data.csv")

    return dataframe