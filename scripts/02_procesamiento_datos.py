import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DATA_PATH,
    PROCESSED_DATA_PATH,
    WINDOW_SIZE,
    HORIZON,
    TRAIN_START,
    TRAIN_END,
    VALIDATION_START,
    TEST_START,
    TEST_END,
    FEATURES_TO_EXCLUDE,
    NORMALIZATION_METHOD,
)
from src.utils.preprocessing import create_sliding_windows


# Cargar el dataset combinado
combined_data = pd.read_csv(DATA_PATH / "combined_data.csv", index_col="Date", parse_dates=True)

# Transformar volumen (log1p)
volume_columns = [col for col in combined_data.columns if col.endswith("_Volume")]
for column in volume_columns:
    combined_data[column] = np.log1p(combined_data[column])

# Crear ventanas deslizantes
X, y, y_params = create_sliding_windows(
    combined_data,
    window_size=WINDOW_SIZE,
    horizon=HORIZON,
    normalization_method=NORMALIZATION_METHOD,
    exclude_columns=FEATURES_TO_EXCLUDE,
)

# Extraer índice temporal para filtrado
if HORIZON == 1:
    dates = combined_data.index[WINDOW_SIZE:]
else:
    dates = combined_data.index[WINDOW_SIZE + HORIZON:]

# Dividir por conjuntos usando fechas
train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
val_mask = (dates > TRAIN_END) & (dates < TEST_START)
test_mask = (dates >= TEST_START) & (dates <= TEST_END)

X_train, y_train, y_train_params = X[train_mask], y[train_mask], [y_params[i] for i in range(len(train_mask)) if train_mask[i]]
X_val, y_val, y_val_params = X[val_mask], y[val_mask], [y_params[i] for i in range(len(val_mask)) if val_mask[i]]
X_test, y_test, y_test_params = X[test_mask], y[test_mask], [y_params[i] for i in range(len(test_mask)) if test_mask[i]]

# Guardar en carpeta procesada
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
np.save(PROCESSED_DATA_PATH / "X_train.npy", X_train)
np.save(PROCESSED_DATA_PATH / "y_train.npy", y_train)
np.save(PROCESSED_DATA_PATH / "X_val.npy", X_val)
np.save(PROCESSED_DATA_PATH / "y_val.npy", y_val)
np.save(PROCESSED_DATA_PATH / "X_test.npy", X_test)
np.save(PROCESSED_DATA_PATH / "y_test.npy", y_test)

# Guardar los parámetros de escalado del target
import pickle
with open(PROCESSED_DATA_PATH / "y_train_params.pkl", "wb") as f:
    pickle.dump(y_train_params, f)
with open(PROCESSED_DATA_PATH / "y_val_params.pkl", "wb") as f:
    pickle.dump(y_val_params, f)
with open(PROCESSED_DATA_PATH / "y_test_params.pkl", "wb") as f:
    pickle.dump(y_test_params, f)