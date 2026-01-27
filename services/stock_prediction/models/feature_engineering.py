import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Lag features
    df["close_lag1"] = df["Close"].shift(1)
    df["close_lag2"] = df["Close"].shift(2)
    df["close_lag3"] = df["Close"].shift(3)

    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # Daily returns
    df['Return'] = df['Close'].pct_change()

    # Volume moving average
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()

    # Drop rows with NaN
    df = df.dropna()

    return df
