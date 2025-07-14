from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import yfinance as yf

from ..config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def download_yahoo_data(
    tickers: Union[str, Iterable[str]],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    save: bool = True,
    destination_directory: Optional[Path] = None,
) -> pd.DataFrame:
    """Descarga datos de Yahoo Finance para uno o varios tickers y los combina.

    Las columnas resultantes estarán prefijadas por el nombre del ticker.
    Si ``save`` es ``True`` el DataFrame combinado se guarda en
    ``RAW_DATA_PATH/combined_data.csv``.
    """

    tickers_list = [tickers] if isinstance(tickers, str) else list(tickers)

    data = yf.download(
        tickers_list,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No se encontraron datos para {tickers_list}.")

    if isinstance(data.columns, pd.MultiIndex):
        # Asegurar que nivel 0 = ticker, nivel 1 = variable
        if data.columns.names != ["Ticker", "Attributes"]:
            data.columns = data.columns.swaplevel()

        # Convertir columnas MultiIndex a columnas simples
        selected_cols = []
        for ticker in tickers_list:
            for attribute in ["Close", "Volume"]:
                if (ticker, attribute) in data.columns:
                    # Excluir volumen si es todo ceros
                    if attribute == "Volume" and data[(ticker, attribute)].sum() == 0:
                        continue
                    selected_cols.append((ticker, attribute))
        selected = data[selected_cols]
        selected.columns = [f"{ticker}_{attribute.replace(' ', '')}" for ticker, attribute in selected.columns]
        data = selected
    else:
        raise ValueError("Se esperaba un MultiIndex en las columnas descargadas.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame descargado no es DatetimeIndex.")

    data.index.name = "Date"

    if save:
        directory = destination_directory or RAW_DATA_PATH
        file_path = directory / "raw_price_data.csv"
        save_dataframe_to_csv(data, file_path)

    return data

def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: Path, include_index: bool = True) -> None:
    # Guarda un DataFrame en formato CSV asegurando que el índice es la fecha

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Validar y nombrar el índice
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")
    dataframe.index.name = "Date"

    # Si file_path es un directorio o no termina en .csv, usar nombre por defecto
    if file_path.suffix != ".csv":
        ticker = dataframe.columns[0] if dataframe.columns.size == 1 else "output"
        file_path = file_path / f"{ticker}.csv"

    dataframe.to_csv(file_path, index=include_index)
