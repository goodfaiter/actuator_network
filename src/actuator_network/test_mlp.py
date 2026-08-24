"""Run MLP inference on test MCAPs."""

import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch, process_inputs_time_series

DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_mlp_latest.pt"


def run_mlp_inference(
    model_path: str,
    mcap_file_paths: list[str],
    data_freq: int,
) -> list[str]:
    """Run MLP inference on the given MCAPs and write prediction MCAPs.

    Args:
        model_path: Path to the saved TorchScript model.
        mcap_file_paths: List of input MCAP files.
        data_freq: Frequency in Hz used during preprocessing.

    Returns:
        List of output file paths.
    """
    device = torch.device("cpu")

    print("Loading MLP model...")
    model = torch.jit.load(model_path, map_location=device)

    stride = int(model.stride.item())
    num_hist = int(model.history_size.item())
    prediction = bool(model.prediction_mode.item())
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
        inputs = process_inputs_time_series(
            data_tensor[:, input_indices], history_size=num_hist, stride=stride, prediction=prediction
        )
        # Flatten windows for the MLP
        inputs = inputs.view(inputs.shape[0], -1)

        # Prepare prediction columns
        for col in output_cols:
            data_df_extrapolated[col + "_predicted"] = 0.0

        with torch.no_grad():
            preds = model(inputs)

        offset = (num_hist - 1) * stride
        for i, col in enumerate(output_cols):
            data_df_extrapolated.loc[
                data_df_extrapolated.index[offset : offset + preds.shape[0]],
                col + "_predicted",
            ] = preds[:, i].numpy()

        output_path = mcap_file_path.replace(".mcap", "_mlp_predicted.mcap")
        data_df_to_mcap(data_df_extrapolated, output_path)
        print(f"  wrote {output_path}")
        output_paths.append(output_path)

    return output_paths


def main():
    # Configuration
    data_freq = 80  # Hz, must match training
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_19/rosbag2_2026_08_19-12_40_03_0.mcap",
    ]

    run_mlp_inference(DEFAULT_MODEL_PATH, mcap_file_paths, data_freq)


if __name__ == "__main__":
    main()
