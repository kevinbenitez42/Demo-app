"""Independent adaptation of Backtesting.ipynb; never executes the notebook."""
import numpy as np
import pandas as pd

STRATEGIES = {
    "Buy and hold": ("daily_return", "buy_hold_index"),
    "MA crossover": ("strategy_return", "strategy_index"),
    "Inverse volatility": ("inverse_vol_return", "inverse_vol_index"),
}


def run_backtest(close, short_window=21, long_window=200):
    if not 1 <= short_window < long_window:
        raise ValueError("The short moving average must be shorter than the long moving average.")
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    if len(close) < max(long_window + 2, 23):
        raise ValueError("Not enough prices for these windows. Choose more history or a shorter long moving average.")
    if not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise ValueError("Prices must be finite positive values.")
    frame = close.rename("Close").to_frame()
    frame["daily_return"] = close.pct_change().fillna(0.0)
    frame["short_ma"] = close.rolling(short_window).mean()
    frame["long_ma"] = close.rolling(long_window).mean()
    frame["signal"] = (frame["short_ma"] > frame["long_ma"]).astype(float)
    frame["position"] = frame["signal"].shift(1).fillna(0.0)
    frame["strategy_return"] = frame["position"] * frame["daily_return"]
    frame["rolling_vol_21d"] = frame["daily_return"].rolling(21).std() * np.sqrt(252)
    inverse = 1 / frame["rolling_vol_21d"].replace(0, np.nan)
    frame["inverse_vol_position"] = (inverse / inverse.expanding().mean()).clip(0, 1.5).shift(1).fillna(0.0)
    frame["inverse_vol_return"] = frame["inverse_vol_position"] * frame["daily_return"]
    rows = []
    for label, (return_column, index_column) in STRATEGIES.items():
        daily = frame[return_column]
        equity = (1 + daily).cumprod()
        frame[index_column] = equity
        drawdown = equity / equity.cummax() - 1
        rows.append({"Strategy": label, "Total return (%)": (equity.iloc[-1] - 1) * 100,
                     "Annualized volatility (%)": daily.std() * np.sqrt(252) * 100,
                     "Max drawdown (%)": drawdown.min() * 100})
    return frame, pd.DataFrame(rows).set_index("Strategy")
