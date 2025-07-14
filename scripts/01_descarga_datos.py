import pandas as pd
from pathlib import Path
import sys

# Añadir la raíz del proyecto al path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_START, TEST_END
from src.config import LONG_WINDOW
from src.utils.data_loading import download_yahoo_data
from src.utils.technical_indicators import compute_technical_indicators

from src.config import TICKERS, TARGET_ASSET


from datetime import timedelta

# Calcular el verdadero inicio de descarga teniendo en cuenta la ventana más larga
window_buffer = max(34, LONG_WINDOW)
download_start = pd.to_datetime(TRAIN_START) - timedelta(days=2 * window_buffer)

# Descargar precios históricos
price_data = download_yahoo_data(TICKERS, start_date=download_start, end_date=TEST_END)
price_data.to_csv(RAW_DATA_PATH / "raw_price_data.csv")

# Calcular indicadores técnicos sobre la serie objetivo
indicators_df = compute_technical_indicators(price_data.copy(), tickers=TICKERS)

# Unir indicadores técnicos a los datos combinados
combined_data = pd.concat([price_data, indicators_df], axis=1)

if not isinstance(combined_data.index, pd.DatetimeIndex):
    combined_data.set_index(price_data.index, inplace=True)

combined_data = combined_data[combined_data.index >= pd.to_datetime(TRAIN_START)]

# Ordenar columnas por nombre de activo (alfabéticamente)
combined_data = combined_data.reindex(sorted(combined_data.columns), axis=1)

# Guardar datos combinados procesados
combined_data.to_csv(PROCESSED_DATA_PATH / "combined_data.csv")