"""Data loading helpers for parallel MCAP processing."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import (
    pandas_to_torch,
    process_inputs_time_series,
    process_outputs_time_series,
)
from actuator_network.helpers.rnn_pipeline import make_contiguous_chunks


def process_mcap_file(
    mcap_file_path: str,
    freq: int,
    input_cols: list[str],
    output_cols: list[str],
    history_size: int,
    stride: int,
    prediction: bool,
    write_processed: bool = True,
    rnn_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and process a single MCAP file into input/output tensors.

    All tensor operations are performed on CPU so the result can be safely returned
    from a multiprocessing worker.

    Args:
        mcap_file_path: Path to the input MCAP file.
        freq: Target resampling frequency in Hz.
        input_cols: Columns to use as model inputs.
        output_cols: Columns to use as model outputs.
        history_size: Number of timesteps in each input window/sequence.
        stride: Stride between consecutive history steps.
        prediction: Whether to shift outputs for prediction mode.
        write_processed: If True, write the processed DataFrame back to an MCAP file.
        rnn_mode: If True, split the tensor into contiguous chunks for stateful RNN training.

    Returns:
        Tuple of (input_tensor, output_tensor) on CPU.
    """
    data_df = read_mcap_to_dataframe(mcap_file_path)
    data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
    # Remove duplicate timestamps by keeping the first occurrence
    data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
    process_dataframe(data_df_extrapolated)

    if write_processed:
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_processed.mcap"))

    col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device="cpu")
    input_indices = [col_names.index(col) for col in input_cols]
    output_indices = [col_names.index(col) for col in output_cols]

    if rnn_mode:
        inputs = make_contiguous_chunks(data_tensor[:, input_indices], history_size)
        outputs = make_contiguous_chunks(data_tensor[:, output_indices], history_size)
    else:
        inputs = process_inputs_time_series(
            data_tensor[:, input_indices],
            stride=stride,
            history_size=history_size,
            prediction=prediction,
        )
        outputs = process_outputs_time_series(
            data_tensor[:, output_indices],
            stride=stride,
            history_size=history_size,
        )

    return inputs, outputs


def _process_mcap_file_kwargs(kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Unpackable wrapper for ProcessPoolExecutor.

    Returns NumPy arrays to avoid PyTorch's inter-process shared-memory mechanism,
    which can fail or deadlock inside forked workers.
    """
    inputs, outputs = process_mcap_file(**kwargs)
    return inputs.numpy(), outputs.numpy()


def load_mcap_files_parallel(
    mcap_file_paths: list[str],
    freq: int,
    input_cols: list[str],
    output_cols: list[str],
    history_size: int,
    stride: int,
    prediction: bool,
    rnn_mode: bool = False,
    max_workers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process multiple MCAP files in parallel and concatenate the resulting tensors.

    Args:
        mcap_file_paths: List of paths to input MCAP files.
        freq: Target resampling frequency in Hz.
        input_cols: Columns to use as model inputs.
        output_cols: Columns to use as model outputs.
        history_size: Number of timesteps in each input window/sequence.
        stride: Stride between consecutive history steps.
        prediction: Whether to shift outputs for prediction mode.
        rnn_mode: If True, split the tensor into contiguous chunks for stateful RNN training.
        max_workers: Maximum number of parallel workers. None uses all CPUs.

    Returns:
        Tuple of concatenated (input_tensor, output_tensor) on CPU.
    """
    kwargs_list = [
        {
            "mcap_file_path": path,
            "freq": freq,
            "input_cols": input_cols,
            "output_cols": output_cols,
            "history_size": history_size,
            "stride": stride,
            "prediction": prediction,
            "write_processed": True,
            "rnn_mode": rnn_mode,
        }
        for path in mcap_file_paths
    ]

    # Limit PyTorch to a single CPU thread before forking. PyTorch's internal
    # thread pools are not fork-safe and can deadlock in worker processes.
    # The original value is restored after loading.
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_mcap_file_kwargs, kwargs_list))
    finally:
        torch.set_num_threads(torch_threads)

    all_inputs_np = np.concatenate([result[0] for result in results], axis=0)
    all_outputs_np = np.concatenate([result[1] for result in results], axis=0)
    return torch.from_numpy(all_inputs_np), torch.from_numpy(all_outputs_np)
