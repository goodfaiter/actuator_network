"""Data loading helpers for parallel MCAP processing."""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
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
    write_processed: bool = True,
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
        write_processed: If True, write the processed DataFrame back to an MCAP file.
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
            "write_processed": write_processed,
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


def _process_mcap_dataframe(mcap_file_path: str, freq: int) -> pd.DataFrame:
    """Read and process a single MCAP file into a DataFrame."""
    data_df = read_mcap_to_dataframe(mcap_file_path)
    data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
    # Remove duplicate timestamps by keeping the first occurrence
    data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
    process_dataframe(data_df_extrapolated)
    return data_df_extrapolated


def _dataframe_cache_path(mcap_file_path: str, freq: int, cache_dir: str) -> str:
    """Return the parquet cache path for a processed MCAP DataFrame.

    The cache key includes the absolute file path, file modification time, and
    target frequency so that editing or moving a source MCAP invalidates the
    cached copy.
    """
    mcap_file_path = os.path.abspath(mcap_file_path)
    mtime = os.path.getmtime(mcap_file_path)
    key = hashlib.sha256(f"{mcap_file_path}:{mtime}:{freq}".encode()).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(mcap_file_path))[0]
    safe_base = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in base)
    return os.path.join(cache_dir, f"{safe_base}_{key}.parquet")


def _load_or_process_dataframe(mcap_file_path: str, freq: int, cache_dir: str) -> pd.DataFrame:
    """Load a processed DataFrame from cache or process and cache it."""
    cache_path = _dataframe_cache_path(mcap_file_path, freq, cache_dir)
    if os.path.isfile(cache_path):
        return pd.read_parquet(cache_path)

    df = _process_mcap_dataframe(mcap_file_path, freq)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def _load_or_process_dataframe_kwargs(kwargs: dict) -> pd.DataFrame:
    """Unpackable wrapper for ProcessPoolExecutor."""
    return _load_or_process_dataframe(kwargs["mcap_file_path"], kwargs["freq"], kwargs["cache_dir"])


def load_mcap_dataframes_parallel(
    mcap_file_paths: list[str],
    freq: int,
    max_workers: int | None = None,
) -> list[pd.DataFrame]:
    """Load and process multiple MCAP files in parallel, returning DataFrames.

    Args:
        mcap_file_paths: List of paths to input MCAP files.
        freq: Target resampling frequency in Hz.
        max_workers: Maximum number of parallel workers. None uses all CPUs.

    Returns:
        List of processed DataFrames, one per input file.
    """
    # Limit PyTorch to a single CPU thread before forking. PyTorch's internal
    # thread pools are not fork-safe and can deadlock in worker processes.
    # The original value is restored after loading.
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            dfs = list(
                executor.map(
                    _process_mcap_dataframe,
                    mcap_file_paths,
                    [freq] * len(mcap_file_paths),
                )
            )
    finally:
        torch.set_num_threads(torch_threads)

    return dfs


def load_mcap_dataframes_parallel_cached(
    mcap_file_paths: list[str],
    freq: int,
    cache_dir: str = "/workspace/data/cache/processed_dataframes",
    max_workers: int | None = None,
) -> list[pd.DataFrame]:
    """Load and process multiple MCAP files, caching processed DataFrames on disk.

    The first call for a given file parses the MCAP, extrapolates, and writes a
    parquet cache. Subsequent calls (including separate agent processes) load
    directly from the cache. The cache is invalidated automatically when the
    source MCAP file is modified.

    Args:
        mcap_file_paths: List of paths to input MCAP files.
        freq: Target resampling frequency in Hz.
        cache_dir: Directory where parquet cache files are stored.
        max_workers: Maximum number of parallel workers. None uses all CPUs.

    Returns:
        List of processed DataFrames, one per input file.
    """
    os.makedirs(cache_dir, exist_ok=True)

    kwargs_list = [
        {
            "mcap_file_path": path,
            "freq": freq,
            "cache_dir": cache_dir,
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
            dfs = list(executor.map(_load_or_process_dataframe_kwargs, kwargs_list))
    finally:
        torch.set_num_threads(torch_threads)

    return dfs
