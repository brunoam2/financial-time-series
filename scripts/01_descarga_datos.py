from pathlib import Path
from datetime import timedelta
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loading import download_price_data
from src.utils.technical_indicators import compute_all_technical_indicators

from src.config import (
    DATA_PATH,
    TRAIN_START,
    TEST_END,
    LONG_WINDOW,
    TICKERS,
    TARGET_COLUMN
)

EXCLUDED_TICKERS = ["^VIX", "DX-Y.NYB"]  # Añade aquí cualquier ticker que quieras excluir del cálculo de indicadores técnicos

download_start = pd.to_datetime(TRAIN_START) - timedelta(days=2 * max(34, LONG_WINDOW))

# Descargar precios históricos
price_data = download_price_data(TICKERS, start_date=download_start, end_date=TEST_END, excluded_tickers=EXCLUDED_TICKERS)

# Determinar ticker objetivo
target_ticker = TARGET_COLUMN.split("_")[0]

tickers_to_process = [t for t in TICKERS if t not in EXCLUDED_TICKERS]

# Calcular indicadores técnicos
indicators_df = compute_all_technical_indicators(price_data.copy(), tickers=tickers_to_process)

combined_data = pd.concat([price_data, indicators_df], axis=1)
combined_data = combined_data[combined_data.index >= pd.to_datetime(TRAIN_START)]

# Ordenar columnas por nombre de activo (alfabéticamente)
combined_data = combined_data.reindex(sorted(combined_data.columns), axis=1)

# Guardar datos combinados procesados
combined_data.to_csv(DATA_PATH / "combined_data.csv")