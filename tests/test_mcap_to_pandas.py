import os

import pandas as pd

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe

TEST_MCAP = "/workspace/tests/test.mcap"

EXPECTED_TOPICS = [
    "/desired_position_rad",
    "/measured_position_rad",
    "/measured_velocity_rad_per_sec",
    "/weight_kg",
    "/bota/wrench_N_and_Nm",
]

EXPECTED_COLUMNS = [
    "measured_position_rad_data",
    "measured_velocity_rad_per_sec_data",
    "bota_wrench_N_and_Nm_force_x",
    "bota_wrench_N_and_Nm_force_y",
    "bota_wrench_N_and_Nm_force_z",
    "bota_wrench_N_and_Nm_torque_x",
    "bota_wrench_N_and_Nm_torque_y",
    "bota_wrench_N_and_Nm_torque_z",
    "weight_kg_data",
    "desired_position_rad_data",
]


def test_read_mcap_to_dataframe_smoke():
    """Reading the test MCAP should return a DataFrame with the expected columns."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    df = read_mcap_to_dataframe(TEST_MCAP)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert list(df.columns) == EXPECTED_COLUMNS


def test_read_mcap_to_dataframe_sorted_datetime_index():
    """The returned DataFrame should have a sorted DatetimeIndex."""
    df = read_mcap_to_dataframe(TEST_MCAP)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_read_mcap_to_dataframe_uses_default_topics():
    """Default topic filtering should return all expected topic-derived columns."""
    df = read_mcap_to_dataframe(TEST_MCAP)

    for col in EXPECTED_COLUMNS:
        assert col in df.columns
