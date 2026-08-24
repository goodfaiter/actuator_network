"""Run M5 + Transformer physics-coupled inference on test MCAPs one timestep at a time."""

import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch

DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_m5_transformer_latest.pt"


def run_m5_transformer_inference(
    model_path: str,
    mcap_file_paths: list[str],
    data_freq: int,
) -> list[str]:
    """Run M5 + Transformer inference on the given MCAPs and write prediction MCAPs.

    Inference is performed one timestep at a time with a sliding history window,
    matching the intended deployment pattern and keeping memory usage low.

    Args:
        model_path: Path to the saved TorchScript model.
        mcap_file_paths: List of input MCAP files.
        data_freq: Frequency in Hz used during preprocessing.

    Returns:
        List of output file paths.
    """
    device = torch.device("cpu")

    print("Loading M5 + Transformer model...")
    model = torch.jit.load(model_path, map_location=device)

    stride = int(model.stride.item())
    num_hist = int(model.history_size.item())
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
        features = data_tensor[:, input_indices]

        # Prepare prediction columns
        for col in output_cols:
            data_df_extrapolated[col + "_predicted"] = 0.0

        # Run inference one timestep at a time through the MCAP.
        num_samples = features.shape[0]
        window_span = (num_hist - 1) * stride + 1
        predictions = torch.zeros((num_samples, len(output_cols)))
        window_offsets = torch.arange(num_hist, device=device) * stride

        for t in range(window_span - 1, num_samples):
            indices = t - (window_span - 1) + window_offsets
            window = features[indices].unsqueeze(0)  # [1, History, Feature]
            with torch.no_grad():
                pred = model(window)
            predictions[t, :] = pred[0, 0, :]

        for i, col in enumerate(output_cols):
            data_df_extrapolated[col + "_predicted"] = predictions[:, i].numpy()

        output_path = mcap_file_path.replace(".mcap", "_m5_transformer_predicted.mcap")
        data_df_to_mcap(data_df_extrapolated, output_path)
        print(f"  wrote {output_path}")
        output_paths.append(output_path)

    return output_paths


def main():
    # Configuration
    data_freq = 200  # Hz, must match training
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-11_58_32_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_18_38_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_34_43_0.mcap",  # strong spring, mixed 200Hz
    ]

    run_m5_transformer_inference(DEFAULT_MODEL_PATH, mcap_file_paths, data_freq)


if __name__ == "__main__":
    main()
