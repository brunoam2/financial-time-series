from pathlib import Path

# ===================== Configuración de los activos =====================
TICKERS = ["SPY", "GLD", "TLT"]
TARGET_COLUMN = "SPY_Close"

# ===================== Configuración de fechas =====================
TRAIN_START = "2005-06-01"
TRAIN_END = "2020-12-31"
VALIDATION_START = "2021-01-01"
VALIDATION_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# ===================== Configuración del experimento =====================
WINDOW_SIZE = 60
HORIZON = 15
MODEL_TYPE = "lstm"
SEED = 42

# Características seleccionadas para el entrenamiento. Si se deja en ``None``
# se utilizan todas las columnas disponibles en los datos procesados.
FEATURES_TO_EXCLUDE: list[str] | None = None

# ===================== Configuración de entrenamiento =====================
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.0

# ===================== Parámetros para indicadores técnicos =====================
SHORT_WINDOW = 10
LONG_WINDOW = 30

# ===================== Definición de rutas =====================
# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_PATH = BASE_DIR / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

# Directorios de resultados
RESULTS_PATH = BASE_DIR / "results"
FIGURES_PATH = RESULTS_PATH / "figures"
PREDICTIONS_PATH = RESULTS_PATH / "predictions"
METRICS_PATH = RESULTS_PATH / "metrics"

# Directorio para modelos entrenados
MODEL_PATH = RESULTS_PATH / "trained_models"


