import numpy as np
import pandas as pd
import pytest

from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe


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


def test_process_dataframe_derives_dt_from_index():
    """The derivative should use the timestep implied by the resampled index spacing.

    A 200 Hz index implies dt = 5 ms; the old hardcoded value was 1/80 s, which
    would have scaled the acceleration by 80/200.
    """
    timestamps = pd.to_timedelta(np.arange(6) * 5.0, unit="ms")
    df = pd.DataFrame(
        {
            "desired_position_rad_data": np.zeros(6),
            "measured_position_rad_data": np.zeros(6),
            "measured_velocity_rad_per_sec_data": np.linspace(0.0, 1.0, 6),
            "weight_kg_data": np.full(6, 0.03),
            "bota_wrench_N_and_Nm_torque_z": np.full(6, 0.011),
        },
        index=timestamps,
    )

    result = process_dataframe(df)

    # Velocity ramps by 0.2 rad/s per sample; dt = 5 ms -> 40 rad/s^2 (times radius).
    expected_acceleration = np.full(6, (0.2 / 0.005) * 0.011)
    expected_acceleration[0] = 0.0  # first difference is NaN -> 0
    np.testing.assert_allclose(
        result["calculated_acceleration_meter_per_sec2_data"].to_numpy(),
        expected_acceleration,
    )
    assert result["tendon_bota_force_newton_data"].nunique() == 1
