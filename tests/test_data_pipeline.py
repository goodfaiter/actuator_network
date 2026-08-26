import os
import tempfile

import pandas as pd
import torch

from actuator_network.helpers.data_pipeline import (
    _dataframe_cache_path,
    load_mcap_dataframes_parallel_cached,
    load_mcap_files_parallel,
    process_mcap_file,
)

TEST_MCAP = "/workspace/tests/test.mcap"
INPUT_COLS = ["desired_position_rad_data", "measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
OUTPUT_COLS = ["load_newton_data"]


def test_process_mcap_file_returns_cpu_tensors():
    """process_mcap_file should return CPU tensors with the expected shapes."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    inputs, outputs = process_mcap_file(
        mcap_file_path=TEST_MCAP,
        freq=80,
        input_cols=INPUT_COLS,
        output_cols=OUTPUT_COLS,
        history_size=10,
        stride=1,
        prediction=False,
        write_processed=False,
    )

    assert isinstance(inputs, torch.Tensor)
    assert isinstance(outputs, torch.Tensor)
    assert inputs.device.type == "cpu"
    assert outputs.device.type == "cpu"
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[1:] == (10, len(INPUT_COLS))
    assert outputs.shape[1:] == (1, len(OUTPUT_COLS))


def test_load_mcap_files_parallel_matches_serial():
    """Parallel loading should produce the same total sample count as serial loading."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    # Use the same file twice to exercise the parallel path with multiple tasks.
    paths = [TEST_MCAP, TEST_MCAP]

    serial_inputs = []
    serial_outputs = []
    for path in paths:
        inputs, outputs = process_mcap_file(
            mcap_file_path=path,
            freq=80,
            input_cols=INPUT_COLS,
            output_cols=OUTPUT_COLS,
            history_size=10,
            stride=1,
            prediction=False,
            write_processed=False,
        )
        serial_inputs.append(inputs)
        serial_outputs.append(outputs)

    parallel_inputs, parallel_outputs = load_mcap_files_parallel(
        mcap_file_paths=paths,
        freq=80,
        input_cols=INPUT_COLS,
        output_cols=OUTPUT_COLS,
        history_size=10,
        stride=1,
        prediction=False,
        write_processed=False,
        max_workers=2,
    )

    assert parallel_inputs.shape[0] == sum(inp.shape[0] for inp in serial_inputs)
    assert parallel_outputs.shape[0] == sum(out.shape[0] for out in serial_outputs)


def test_dataframe_cache_path_is_stable_and_unique():
    """Cache path should be deterministic and change when mtime/freq changes."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = _dataframe_cache_path(TEST_MCAP, freq=80, cache_dir=tmpdir)
        path_b = _dataframe_cache_path(TEST_MCAP, freq=80, cache_dir=tmpdir)
        path_c = _dataframe_cache_path(TEST_MCAP, freq=200, cache_dir=tmpdir)

        assert path_a == path_b
        assert path_a != path_c
        assert path_a.endswith(".parquet")


def test_load_mcap_dataframes_parallel_cached_creates_cache():
    """The cached loader should process MCAPs and write parquet cache files."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        dfs = load_mcap_dataframes_parallel_cached([TEST_MCAP], freq=80, cache_dir=tmpdir)
        assert len(dfs) == 1
        assert isinstance(dfs[0], pd.DataFrame)
        assert not dfs[0].empty

        cache_path = _dataframe_cache_path(TEST_MCAP, freq=80, cache_dir=tmpdir)
        assert os.path.isfile(cache_path)


def test_load_mcap_dataframes_parallel_cached_uses_existing_cache():
    """When a cache file exists, the loader should return it without reprocessing."""
    assert os.path.isfile(TEST_MCAP), f"Test MCAP not found: {TEST_MCAP}"

    with tempfile.TemporaryDirectory() as tmpdir:
        dfs_first = load_mcap_dataframes_parallel_cached([TEST_MCAP], freq=80, cache_dir=tmpdir)
        cache_path = _dataframe_cache_path(TEST_MCAP, freq=80, cache_dir=tmpdir)
        cache_mtime = os.path.getmtime(cache_path)

        dfs_second = load_mcap_dataframes_parallel_cached([TEST_MCAP], freq=80, cache_dir=tmpdir)

        assert len(dfs_first) == len(dfs_second) == 1
        pd.testing.assert_frame_equal(dfs_first[0], dfs_second[0], check_freq=False)
        assert os.path.getmtime(cache_path) == cache_mtime
