# ========================= IMPORTACIONES =========================
from pathlib import Path
from datetime import timedelta
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loading import download_price_data
from src.utils.technical_indicators import compute_technical_indicators

from src.config import (
    DATA_PATH,
    TRAIN_START,
    TEST_END,
    LONG_WINDOW,
    TICKERS,
)
 # ================================================================

# Se calcula la fecha de inicio para la descarga de datos (teniendo en cuenta el largo de la ventana de observaciones)
download_start = pd.to_datetime(TRAIN_START) - timedelta(days=2 * max(34, LONG_WINDOW))

# Se descargan los datos de precios
price_data = download_price_data(TICKERS, start_date=download_start, end_date=TEST_END)

# Se calculan indicadores técnicos
indicators_df = compute_technical_indicators(price_data.copy(), tickers=TICKERS)

# Se concatenan ambos DataFrames
combined_data = pd.concat([price_data, indicators_df], axis=1)
combined_data = combined_data[combined_data.index >= pd.to_datetime(TRAIN_START)]

# Se ordenan las columnas por nombre de activo
combined_data = combined_data.reindex(sorted(combined_data.columns), axis=1)

# Se guardan los datos procesados en csv
combined_data.to_csv(DATA_PATH / "combined_data.csv")