import ta
import pandas as pd
from src.config import SHORT_WINDOW, LONG_WINDOW


def compute_all_technical_indicators(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Calcula todos los indicadores técnicos para múltiples tickers."""
    indicators = pd.DataFrame(index=df.index)
    for ticker in tickers:
        price_col = f"{ticker}_Close"

        indicators[f"{ticker}_Return"] = df[price_col].pct_change(fill_method=None)

        sma_short = ta.trend.SMAIndicator(close=df[price_col], window=SHORT_WINDOW).sma_indicator()
        sma_long = ta.trend.SMAIndicator(close=df[price_col], window=LONG_WINDOW).sma_indicator()

        indicators[f"{ticker}_RelativePriceSMA{SHORT_WINDOW}"] = (df[price_col] - sma_short) / sma_short
        indicators[f"{ticker}_RelativePriceSMA{LONG_WINDOW}"] = (df[price_col] - sma_long) / sma_long
        indicators[f"{ticker}_MA_Crossover"] = (sma_short - sma_long) / sma_long
        indicators[f"{ticker}_Volatility"] = df[price_col].rolling(window=21).std() / df[price_col]
        indicators[f"{ticker}_RSI"] = ta.momentum.RSIIndicator(close=df[price_col], window=14).rsi() / 100

    return indicators