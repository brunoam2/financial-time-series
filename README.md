# Financial Time Series Forecasting

Este proyecto contiene utilidades para descargar datos de activos financieros, procesarlos en forma de ventanas deslizantes y entrenar varios modelos de predicción.

## Estructura básica

1. `scripts/01_descarga_datos.py` descarga los precios y calcula indicadores técnicos.
2. `scripts/02_procesamiento_datos.py` genera las ventanas y divide en conjuntos de entrenamiento, validación y prueba.
3. `scripts/03_entrenamiento_modelo.py` entrena el modelo definido en `src/config.py` y guarda las predicciones.
4. `scripts/04_evaluacion_predicciones.py` compara las predicciones de todos los modelos almacenados.

Los parámetros principales (ventana, horizonte, selección de columnas, etc.) se configuran en `src/config.py`.
Desde la versión actual también se incluye un modelo `cnn` basado en convoluciones 1D, además de mejoras en los modelos recurrentes.
