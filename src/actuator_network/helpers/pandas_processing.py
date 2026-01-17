import pandas as pd

def extrapolate_dataframe(df: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Extrapolate missing data in DataFrame to a fixed frequency F (Hz)"""

    target_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=f'{1000//freq}ms',
        )

    df_extrapolated = pd.DataFrame(index=target_index)

    for col in df.columns:
        df_extrapolated[col] = df[col].reindex(target_index, method="ffill", limit=1).interpolate(method="linear").bfill().ffill()

    df_extrapolated.index = df_extrapolated.index - df_extrapolated.index[0]

    return df_extrapolated