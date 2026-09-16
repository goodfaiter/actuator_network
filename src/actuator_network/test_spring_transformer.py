"""Run inference with the spring-class + force transformer model."""

import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch

# DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_spring_transformer_from_sweep.pt"
DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_spring_transformer_latest.pt"
# DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_spring_transformer_sweep_mtyeemuo_latest.pt"


def _build_inference_window(
    features: torch.Tensor,
    t: int,
    num_hist: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a zero-padded history window for timestep ``t``.

    Early timesteps for which the history would extend before the start of the
    data are included and padded with zeros at the beginning of the window.

    Args:
        features: Input tensor of shape ``(num_samples, feature_dim)``.
        t: Current timestep.
        num_hist: Number of history steps in the window.
        stride: Stride between history samples.
        device: Torch device.

    Returns:
        Window tensor of shape ``(1, num_hist, feature_dim)``.
    """
    window_span = (num_hist - 1) * stride + 1
    window_offsets = torch.arange(num_hist, device=device) * stride
    indices = t - (window_span - 1) + window_offsets
    indices_clamped = indices.clamp_min(0)
    window = features[indices_clamped].clone()
    window[indices < 0] = 0.0
    return window.unsqueeze(0)  # [1, History, Feature]


def run_spring_transformer_inference(
    model_path: str,
    mcap_file_paths: list[str],
) -> list[str]:
    """Run spring/force transformer inference on the given MCAPs.

    Inference is performed one timestep at a time with a sliding history window,
    matching the intended online deployment pattern. The preprocessing frequency
    is read from the model metadata so it always matches the frequency used
    during training.

    Args:
        model_path: Path to the saved TorchScript model.
        mcap_file_paths: List of input MCAP files.

    Returns:
        List of output file paths.
    """
    device = torch.device("cpu")

    output_paths = []
    for mcap_file_path in mcap_file_paths:
        print("Loading spring transformer model...")
        # Reload per file: the model is stateful (spring buffer) and reset() does
        # not survive torch.jit.script, so a fresh instance is used per recording.
        model = torch.jit.load(model_path, map_location=device)

        data_freq = model.metadata["frequency"]
        if data_freq <= 1:
            raise ValueError(f"Checkpoint stores an invalid frequency: {data_freq}")

        stride = model.metadata["stride"]
        num_hist = model.metadata["history_size"]
        input_cols = model.input_columns
        output_cols = model.output_columns
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=data_freq)
        data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
        process_dataframe(data_df_extrapolated)
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
        input_indices = [col_names.index(col) for col in input_cols]
        features = data_tensor[:, input_indices]

        for col in output_cols:
            data_df_extrapolated[col + "_predicted"] = 0.0

        num_samples = features.shape[0]
        predictions = torch.zeros((num_samples, len(output_cols)))

        for t in range(num_samples):
            window = _build_inference_window(features, t, num_hist, stride, device=device)
            with torch.no_grad():
                pred = model(window)
            predictions[t, :] = pred[0, 0, :]

        for i, col in enumerate(output_cols):
            data_df_extrapolated[col + "_predicted"] = predictions[:, i].numpy()

        output_path = mcap_file_path.replace(".mcap", "_spring_transformer_predicted.mcap")
        data_df_to_mcap(data_df_extrapolated, output_path)
        print(f"  wrote {output_path}")
        output_paths.append(output_path)

    return output_paths


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_16_16_0.mcap", # blocked
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_46_40_0.mcap", # strong spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_27_33_0.mcap", # weak spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_19_09_0.mcap", # finger
    ]

    run_spring_transformer_inference(DEFAULT_MODEL_PATH, mcap_file_paths)


if __name__ == "__main__":
    main()
