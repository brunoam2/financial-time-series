# Financial Time Series Forecasting

Este proyecto permite descargar datos de varios activos financieros, procesarlos en forma de ventanas deslizantes y entrenar diferentes modelos para predecir la evolución futura de los precios.

## Instalación

1. Crea un entorno de Python (se recomienda 3.10 o superior).
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Estructura de carpetas

```
financial-time-series/
├── scripts/                 # Programas a ejecutar en orden
│   ├── 01_descarga_datos.py
│   ├── 02_procesamiento_datos.py
│   ├── 03_entrenamiento_modelo.py
│   └── 04_evaluacion_predicciones.py
├── src/
│   ├── config.py            # Archivo de configuración del experimento
│   ├── models/              # Implementaciones de modelos (LSTM, GRU, CNN, ...)
│   ├── pipeline/            # Lógica de entrenamiento
│   └── utils/               # Funciones auxiliares
├── data/                    # Se crea automáticamente para almacenar datos
└── results/                 # Salida de modelos, predicciones y gráficos
```

## Configuración

Todas las variables modificables se encuentran en `src/config.py`.
Aquí puedes ajustar:

- `TICKERS`: activos a descargar.
- `TARGET_COLUMN`: columna objetivo a predecir.
- Fechas `START` y `END` para el período de datos.
- `WINDOW_SIZE` y `HORIZON` para las ventanas y horizonte de predicción.
- `MODEL_TYPE` con el modelo a utilizar (`lstm`, `gru`, `transformer`, `cnn`, `xgboost`).
- Otros parámetros relacionados con entrenamiento y rutas.

Modifica estos valores según tus necesidades antes de ejecutar los scripts.

## Ejecución

Ejecuta los scripts de la carpeta `scripts` en el orden numerado:

```bash
python scripts/01_descarga_datos.py        # Descarga precios e indicadores
python scripts/02_procesamiento_datos.py   # Crea ventanas de entrenamiento
python scripts/03_entrenamiento_modelo.py  # Entrena el modelo elegido
python scripts/04_evaluacion_predicciones.py  # Compara y grafica resultados
```

Tras la ejecución encontrarás los modelos entrenados, las predicciones y las métricas dentro de la carpeta `results/`.
