# ========================= IMPORTACIONES =========================
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DATA_PATH,
    PROCESSED_DATA_PATH,
    FEATURES_TO_EXCLUDE,
)
from src.utils.preprocessing import create_sliding_windows

# ================================================================

# Cargar el dataset combinado
combined_data = pd.read_csv(DATA_PATH / "combined_data.csv", index_col="Date", parse_dates=True)

# Transformar volumen (log1p) para que sea más robusto a valores extremos
volume_columns = [col for col in combined_data.columns if col.endswith("_Volume")]
for column in volume_columns:
    combined_data[column] = np.log1p(combined_data[column])

# Exlcuir columnas antes de crear las ventanas
if FEATURES_TO_EXCLUDE:
    combined_data = combined_data.drop(columns=FEATURES_TO_EXCLUDE, errors="ignore")

# Crear ventanas deslizantes
X, y, y_params = create_sliding_windows(combined_data)

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
np.save(PROCESSED_DATA_PATH / "X.npy", X)
np.save(PROCESSED_DATA_PATH / "y.npy", y)
np.save(PROCESSED_DATA_PATH / "y_params.npy", y_params)
