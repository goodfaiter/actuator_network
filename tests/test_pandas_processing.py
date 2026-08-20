import numpy as np
import pandas as pd
import pytest

from actuator_network.helpers.pandas_processing import extrapolate_dataframe


def test_extrapolate_dataframe_produces_no_nans():
    """Extrapolation should fill gaps and edge NaNs so downstream code sees complete columns."""
    timestamps = pd.to_datetime([0, 13, 25, 40, 55, 80, 120], unit="ms")
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan],
            "b": [10.0, 12.0, np.nan, 14.0, 15.0, np.nan, 18.0],
        },
        index=timestamps,
    )

    result = extrapolate_dataframe(df, freq=80)

    assert result.isna().sum().sum() == 0


def test_extrapolate_dataframe_target_frequency_and_monotonic_index():
    """The output index should be evenly spaced at the requested frequency and start at zero."""
    timestamps = pd.to_datetime([5, 18, 32, 50], unit="ms")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=timestamps)

    result = extrapolate_dataframe(df, freq=80)

    # 80 Hz -> 12.5 ms period; index should start at 0.
    assert result.index.freqstr is not None
    assert pd.Timedelta(result.index.freq) == pd.Timedelta("12.5ms")
    assert result.index[0] == pd.Timedelta(0)
    assert result.index.is_monotonic_increasing


def test_extrapolate_dataframe_interpolates_within_range():
    """Values between known samples should be linearly interpolated."""
    timestamps = pd.to_datetime([0, 25], unit="ms")
    df = pd.DataFrame({"a": [0.0, 10.0]}, index=timestamps)

    result = extrapolate_dataframe(df, freq=80)

    # 80 Hz grid: 0, 12.5, 25 ms. Linear interpolation at 12.5 ms is 5.0.
    assert result.loc[pd.Timedelta("12.5ms"), "a"] == pytest.approx(5.0)
