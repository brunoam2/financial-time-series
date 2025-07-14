from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TRAIN_START,
    TEST_END,
    LONG_WINDOW,
    TICKERS,
)
from src.utils.data_loading import download_yahoo_data
from src.utils.technical_indicators import compute_technical_indicators


from datetime import timedelta

download_start = pd.to_datetime(TRAIN_START) - timedelta(days=2 * max(34, LONG_WINDOW))

# Descargar precios históricos
price_data = download_yahoo_data(TICKERS, start_date=download_start, end_date=TEST_END)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
price_data.to_csv(RAW_DATA_PATH / "raw_price_data.csv")

# Calcular indicadores técnicos sobre la serie objetivo
indicators_df = compute_technical_indicators(price_data.copy(), tickers=TICKERS)

# Unir indicadores técnicos a los datos combinados
combined_data = pd.concat([price_data, indicators_df], axis=1)

combined_data = combined_data[combined_data.index >= pd.to_datetime(TRAIN_START)]

# Ordenar columnas por nombre de activo (alfabéticamente)
combined_data = combined_data.reindex(sorted(combined_data.columns), axis=1)

# Guardar datos combinados procesados
combined_data.to_csv(PROCESSED_DATA_PATH / "combined_data.csv")