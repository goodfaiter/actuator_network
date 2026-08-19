import os
import tempfile

import pandas as pd
from mcap_ros2.reader import read_ros2_messages

from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap


def test_data_df_to_mcap_creates_single_file():
    timestamps = pd.date_range(start="2026-01-01", periods=10, freq="10ms")
    df = pd.DataFrame(
        {
            "desired_position_rad_data": [0.1 * i for i in range(10)],
            "load_newton_data": [1.0 * i for i in range(10)],
        },
        index=timestamps,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.mcap")
        data_df_to_mcap(df, output_path)

        assert os.path.isfile(output_path), "Expected a single MCAP file to be created"

        msgs = list(read_ros2_messages(output_path))
        assert len(msgs) == len(df) * len(df.columns)

        topic_names = {msg.channel.topic for msg in msgs}
        assert topic_names == {"/desired_position_rad_data", "/load_newton_data"}
