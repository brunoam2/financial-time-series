import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_PATH, DATA_PATH, TARGET_COLUMN, WINDOW_SIZE

# Cargar datos procesados
X_train = np.load(PROCESSED_DATA_PATH / "X_train.npy")
y_train = np.load(PROCESSED_DATA_PATH / "y_train.npy")
with open(PROCESSED_DATA_PATH / "y_train_params.pkl", "rb") as f:
    y_train_params = pickle.load(f)

# Cargar conjunto original combinado
combined_data = pd.read_csv(DATA_PATH / "combined_data.csv", index_col="Date", parse_dates=True)

print("Comprobando dimensiones...")
assert X_train.shape[0] == y_train.shape[0] == len(y_train_params), "Número de ventanas, etiquetas y parámetros no coinciden"
assert X_train.shape[1] == WINDOW_SIZE, f"Tamaño de la ventana incorrecto: {X_train.shape[1]} != {WINDOW_SIZE}"

print("✔️ Dimensiones validadas")

# Validar normalización: media≈0 y std≈1 para retornos
print("Comprobando normalización de retornos...")
return_columns = [col for col in combined_data.columns if "Return" in col]
return_indices = [i for i, col in enumerate(combined_data.columns) if col in return_columns]

sample_returns = X_train[:, :, return_indices]
means = sample_returns.mean(axis=(0, 1))
stds = sample_returns.std(axis=(0, 1))

print("Media de retornos normalizados (esperado ≈ 0):", means)
print("Desviación estándar de retornos normalizados (esperado ≈ 1):", stds)

# Validar rango [0, 1] para precios normalizados
print("Comprobando precios normalizados...")
price_columns = [col for col in combined_data.columns if "Close" in col]
price_indices = [i for i, col in enumerate(combined_data.columns) if col in price_columns]

sample_prices = X_train[:, :, price_indices]
min_prices = sample_prices.min(axis=(0, 1))
max_prices = sample_prices.max(axis=(0, 1))

print("Mínimos de precios normalizados (esperado ≥ 0):", min_prices)
print("Máximos de precios normalizados (esperado ≤ 1):", max_prices)

# Comprobación de consistencia entre ventanas e índices originales
print("Verificando alineación de ventanas con los datos originales...")
first_window_raw = combined_data.iloc[0:WINDOW_SIZE].values
first_window_processed = X_train[0]

print("Primera fila de la primera ventana original:")
print(first_window_raw[0])

print("Primera fila de la primera ventana procesada:")
print(first_window_processed[0])

print("✅ Comprobación completada")