"""Run RNN inference on test MCAPs."""

import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch
from actuator_network.helpers.rnn_pipeline import run_stateful_inference

DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_rnn_latest.pt"


def run_rnn_inference(
    model_path: str,
    mcap_file_paths: list[str],
) -> list[str]:
    """Run RNN inference on the given MCAPs and write prediction MCAPs.

    The preprocessing frequency is read from the model metadata so it always
    matches the frequency used during training.

    Args:
        model_path: Path to the saved TorchScript model.
        mcap_file_paths: List of input MCAP files.

    Returns:
        List of output file paths.
    """
    device = torch.device("cpu")

    print("Loading RNN model...")
    model = torch.jit.load(model_path, map_location=device)

    data_freq = model.metadata["frequency"]
    if data_freq <= 1:
        raise ValueError(f"Checkpoint stores an invalid frequency: {data_freq}")

    num_hist = model.metadata["history_size"]
    input_cols = model.input_columns
    output_cols = model.output_columns

    output_paths = []
    for mcap_file_path in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=data_freq)
        # Remove duplicate timestamps by keeping the first occurrence
        data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
        process_dataframe(data_df_extrapolated)
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
        input_indices = [col_names.index(col) for col in input_cols]

        run_stateful_inference(
            model,
            input_data=data_tensor[:, input_indices],
            output_cols=output_cols,
            seq_length=num_hist,
            data_df=data_df_extrapolated,
        )

        output_path = mcap_file_path.replace(".mcap", "_rnn_predicted.mcap")
        data_df_to_mcap(data_df_extrapolated, output_path)
        print(f"  wrote {output_path}")
        output_paths.append(output_path)

    return output_paths


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-11_58_32_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_18_38_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_34_43_0.mcap",  # strong spring, mixed 200Hz
    ]

    run_rnn_inference(DEFAULT_MODEL_PATH, mcap_file_paths)


if __name__ == "__main__":
    main()
