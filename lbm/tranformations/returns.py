import pandas as pd

def get_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change()

def apply_returns(init: float, returns: pd.DataFrame) -> pd.DataFrame:
    result = returns.copy()
    result.iloc[0, 0] = init
    for i in range(1, len(returns)):
        result.iloc[i, 0] = result.iloc[i - 1, 0] * (1 + returns.iloc[i, 0])
    return result