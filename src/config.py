from pathlib import Path

# ===================== Configuración de los activos =====================
TICKERS = ["SPY", "GLD", "TLT"]
TARGET_COLUMN = "SPY_Close"

# ===================== Configuración de fechas =====================
# Periodos de entrenamiento y prueba actualizados para utilizar datos más
# recientes y equilibrar la cantidad de observaciones por conjunto.
START = "2007-01-01"
END = "2024-12-31"

# ===================== Configuración del experimento =====================
# Longitud de ventana y horizonte ajustados para predicciones de medio plazo.
WINDOW_SIZE = 250
HORIZON = 120
MODEL_TYPE = "gru"
SEED = 42

# Características seleccionadas para el entrenamiento. Si se deja en ``None``
# se utilizan todas las columnas disponibles en los datos procesados.
FEATURES_TO_EXCLUDE: list[str] | None = None

# ===================== Configuración de entrenamiento =====================
# Menor paciencia y delta mínima para acelerar la parada y evitar overfitting.
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# ===================== Parámetros para indicadores técnicos =====================
SHORT_WINDOW = 30
LONG_WINDOW = 120

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


