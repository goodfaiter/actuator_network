import os

import torch

from actuator_network.helpers.data_pipeline import load_mcap_files_parallel, process_mcap_file

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
        max_workers=2,
    )

    assert parallel_inputs.shape[0] == sum(inp.shape[0] for inp in serial_inputs)
    assert parallel_outputs.shape[0] == sum(out.shape[0] for out in serial_outputs)
