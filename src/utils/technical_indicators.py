from src.config import SHORT_WINDOW, LONG_WINDOW
import ta
import pandas as pd
import numpy as np

def compute_technical_indicators(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Añade indicadores técnicos para múltiples tickers al DataFrame usando la librería 'ta'.

    Incluye:
    - Precio relativo a media móvil simple (corta y larga)
    - RSI
    - Volatilidad histórica (rolling std)
    - MACD
    - Relación entre medias móviles
    """
    all_indicators = pd.DataFrame(index=df.index)

    for ticker in tickers:
        price_column = next((col for col in df.columns if col.lower().startswith(ticker.lower()) and col.lower().endswith("close")), None)
        if price_column is None:
            raise ValueError(f"No se encontró una columna de cierre para el ticker {ticker}.")
        prefix = f"{ticker}_"
        indicators_df = pd.DataFrame(index=df.index)

        # Retorno logarítmico
        indicators_df[f"{prefix}LogReturn"] = np.log(df[price_column] / df[price_column].shift(1))

        sma_short = ta.trend.SMAIndicator(close=df[price_column], window=SHORT_WINDOW).sma_indicator()
        indicators_df[f"{prefix}RelativePriceSMA{SHORT_WINDOW}"] = (df[price_column] - sma_short) / sma_short

        sma_long = ta.trend.SMAIndicator(close=df[price_column], window=LONG_WINDOW).sma_indicator()
        indicators_df[f"{prefix}RelativePriceSMA{LONG_WINDOW}"] = (df[price_column] - sma_long) / sma_long

        indicators_df[f"{prefix}MA_Relative"] = (sma_short - sma_long) / sma_long

        indicators_df[f"{prefix}Volatility"] = df[price_column].rolling(window=SHORT_WINDOW).std() / df[price_column]

        indicators_df[f"{prefix}RSI"] = ta.momentum.RSIIndicator(close=df[price_column], window=SHORT_WINDOW).rsi()

        macd = ta.trend.MACD(close=df[price_column])
        indicators_df[f"{prefix}MACD_diff"] = (macd.macd() - macd.macd_signal()) / df[price_column]
        indicators_df[f"{prefix}MACD_signal"] = macd.macd_signal() / df[price_column]

        all_indicators = pd.concat([all_indicators, indicators_df], axis=1)

    return all_indicators