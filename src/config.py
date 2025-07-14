from pathlib import Path

# ===================== Configuración de los activos =====================
TICKERS = ["SPY", "GLD", "TLT", "^VIX", "DX-Y.NYB"]
TARGET_COLUMN = "SPY_LogReturn"

# ===================== Configuración de fechas =====================
TRAIN_START = "2018-01-01"
TRAIN_END = "2020-12-31"
VALIDATION_START = "2021-01-01"
VALIDATION_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2022-12-31"

# ===================== Configuración del experimento =====================
WINDOW_SIZE = 30
NORMALIZATION_METHOD = "minmax"
MODEL_TYPE = "arima" 
SEED = 42
TEST_SIZE = 0.2
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0

# ===================== Parámetros para indicadores técnicos =====================
SHORT_WINDOW = 10
LONG_WINDOW = 30

# ===================== Definición de rutas =====================
# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_PATH = BASE_DIR / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

# Directorios de resultados
RESULTS_PATH = BASE_DIR / "results"
FIGURES_PATH = RESULTS_PATH / "figures"
PREDICTIONS_PATH = RESULTS_PATH / "predictions"
METRICS_PATH = RESULTS_PATH / "metrics"

# Directorio para modelos entrenados
MODEL_PATH = RESULTS_PATH / "models"


# Parámetros por defecto para la preparación de datos y entrenamiento
SELECTED_VARIABLES = ["Close"]

# Variables seleccionadas para entrenamiento
SELECTED_FEATURES = [
    "GLD_Close",
    "TLT_Volume",
    "TLT_Close",
    "TLT_Volatility",
    "DX-Y.NYB_MACD_signal",
    "DX-Y.NYB_Close",
    "SPY_MACD_signal",
    "DX-Y.NYB_MA_Relative",
    "DX-Y.NYB_RelativePriceSMA30",
    "SPY_MA_Relative"
]
