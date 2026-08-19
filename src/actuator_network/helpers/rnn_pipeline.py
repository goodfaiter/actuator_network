"""Shared helpers for stateful RNN chunking and inference."""

import pandas as pd
import torch


def make_contiguous_chunks(data: torch.Tensor, seq_length: int) -> torch.Tensor:
    """Split a time series tensor into contiguous non-overlapping chunks."""
    num_chunks = data.shape[0] // seq_length
    if num_chunks == 0:
        return torch.empty((0, seq_length, data.shape[-1]), device=data.device)
    return data[: num_chunks * seq_length].view(num_chunks, seq_length, data.shape[-1])


def run_stateful_inference(
    model: torch.jit.ScriptModule,
    input_data: torch.Tensor,
    output_cols: list[str],
    seq_length: int,
    data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run stateful chunk-wise inference and write per-timestep predictions into the DataFrame.

    Args:
        model: Loaded TorchScript model with a `reset()` method.
        input_data: Input tensor of shape (num_timesteps, input_dim).
        output_cols: List of output column names.
        seq_length: Number of timesteps per chunk.
        data_df: DataFrame to write the predicted columns into.

    Returns:
        The DataFrame with populated `*_predicted` columns.
    """
    num_chunks = input_data.shape[0] // seq_length
    if num_chunks == 0:
        return data_df

    for col in output_cols:
        data_df[col + "_predicted"] = 0.0

    if hasattr(model, "reset"):
        model.reset()

    for k in range(num_chunks):
        chunk = input_data[k * seq_length : (k + 1) * seq_length].unsqueeze(0)
        with torch.no_grad():
            pred = model(chunk)  # shape: (1, seq_length, output_dim)
        for i, col in enumerate(output_cols):
            start = k * seq_length
            end = (k + 1) * seq_length
            data_df.loc[data_df.index[start:end], col + "_predicted"] = pred[0, :, i].numpy()

    return data_df
