"""Train a spring-class estimator + force estimator transformer pair jointly.

The spring transformer is trained to estimate a continuous spring coefficient
(0.0 = weak, 0.5 = finger, 1.0 = strong) from a frozen input buffer of
[delta_position, velocity]. The buffer is updated only when |velocity| > 0.1;
otherwise the previous buffer is reused. The force transformer receives
[delta_position, velocity, estimated_spring] and estimates
``tendon_bota_force_newton_data``. Both transformers are trained end-to-end
with a combined loss: force MSE + auxiliary spring MSE.
"""

import pandas as pd
import torch
import torch.nn as nn

from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import (
    normalize_tensor,
    pandas_to_torch,
    process_inputs_time_series,
    process_outputs_time_series,
)
from actuator_network.helpers.torch_model import (
    SpringForceTrainingModel,
    SpringTransformerForceEstimator,
    TorchTransformerModel,
)
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper

OUTPUT_DIR = "/workspace/data/output_data/"

INPUT_COLS = ["measured_position_rad_data", "desired_position_rad_data", "measured_velocity_rad_per_sec_data"]
OUTPUT_COL = "tendon_bota_force_newton_data"
SPRING_COL = "spring_coeff"


def _build_frozen_spring_windows(
    normal_windows: torch.Tensor,
    velocity_idx: int,
    velocity_threshold: float,
) -> torch.Tensor:
    """Build spring windows where the buffer is frozen while |velocity| <= threshold.

    Args:
        normal_windows: Sliding windows of shape [N, H, F].
        velocity_idx: Index of the velocity channel.
        velocity_threshold: Velocity magnitude below which  the buffer freezes.

    Returns:
        Spring windows of the same shape as ``normal_windows``.
    """
    num_samples = normal_windows.size(0)
    spring_windows = normal_windows.clone()
    last_moving_window = torch.zeros_like(normal_windows[0])

    for i in range(num_samples):
        if torch.abs(normal_windows[i, -1, velocity_idx]) > velocity_threshold:
            last_moving_window = normal_windows[i].clone()
        spring_windows[i] = last_moving_window

    return spring_windows


def build_estimated_spring_dataset(
    dataframes: list[pd.DataFrame],
    file_labels: list[tuple[str, float]],
    history_size: int,
    stride: int,
    prediction: bool,
    velocity_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build frozen spring windows, normal force windows, spring targets, and force targets.

    Args:
        dataframes: Processed DataFrames, one per MCAP file.
        file_labels: List of ``(mcap_path, spring_coefficient)`` pairs. The spring
            coefficient is a continuous label (e.g., 0.0 = weak, 0.5 = finger,
            1.0 = strong).
        history_size: Length of each input window.
        stride: Stride between consecutive windows.
        prediction: Whether to shift outputs for prediction mode.
        velocity_threshold: Velocity magnitude below which the spring buffer freezes.
        device: Torch device to place tensors on.

    Returns:
        Tuple of (spring_windows, force_windows, spring_targets, force_targets).
    """
    all_spring_windows = []
    all_force_windows = []
    all_spring_targets = []
    all_force_targets = []

    velocity_idx = INPUT_COLS.index("measured_velocity_rad_per_sec_data")

    for df, (_, spring_label) in zip(dataframes, file_labels):
        df[SPRING_COL] = spring_label

        col_names, data_tensor = pandas_to_torch(df, device="cpu")
        input_indices = [col_names.index(col) for col in INPUT_COLS]
        output_idx = col_names.index(OUTPUT_COL)
        spring_idx = col_names.index(SPRING_COL)

        features = data_tensor[:, input_indices]
        normal_windows = process_inputs_time_series(
            features,
            history_size=history_size,
            stride=stride,
            prediction=prediction,
        )
        spring_windows = _build_frozen_spring_windows(
            normal_windows,
            velocity_idx=velocity_idx,
            velocity_threshold=velocity_threshold,
        )

        spring_targets = process_outputs_time_series(
            data_tensor[:, spring_idx].unsqueeze(1),
            stride=stride,
            history_size=history_size,
        )
        force_targets = process_outputs_time_series(
            data_tensor[:, output_idx].unsqueeze(1),
            stride=stride,
            history_size=history_size,
        )

        all_spring_windows.append(spring_windows)
        all_force_windows.append(normal_windows)
        all_spring_targets.append(spring_targets)
        all_force_targets.append(force_targets)

    return (
        torch.cat(all_spring_windows, dim=0).to(device),
        torch.cat(all_force_windows, dim=0).to(device),
        torch.cat(all_spring_targets, dim=0).to(device),
        torch.cat(all_force_targets, dim=0).to(device),
    )


def make_spring_force_loss(aux_weight: float = 1.0):
    """Return a loss function combining force MSE and auxiliary spring MSE.

    Args:
        aux_weight: Weight for the spring MSE term relative to the force MSE term.

    Returns:
        Callable ``loss_fn(pred, target)`` where both tensors have shape
        ``[Batch, 1, 2]`` with channels ``[force, spring]``.
    """
    criterion = nn.MSELoss()

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        force_loss = criterion(pred[:, :, 0:1], target[:, :, 0:1])
        spring_loss = criterion(pred[:, :, 1:2], target[:, :, 1:2])
        return force_loss + aux_weight * spring_loss

    return loss_fn


def main():
    # Configuration.
    data_freq = 200
    stride = 2
    inference_freq = data_freq // stride
    prediction = False
    history_size = 150
    velocity_threshold = 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_cols = [OUTPUT_COL, SPRING_COL]

    mcap_files: list[tuple[str, float]] = [
        # finger, mixed 200Hz
        ("/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_03_30_0.mcap", 0.5),
        ("/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_52_16_0.mcap", 0.5),
        # weak spring, mixed 200Hz
        ("/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_11_49_0.mcap", 0.0),
        ("/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_15_46_0.mcap", 0.0),
        # strong spring, mixed 200Hz
        ("/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_27_46_0.mcap", 1.0),
        ("/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_31_31_0.mcap", 1.0),
    ]

    # Training knobs.
    aux_weight = 1.0
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 1024
    train_ratio = 0.9

    print("Loading and processing MCAP files...")
    dataframes = load_mcap_dataframes_parallel([path for path, _ in mcap_files], freq=data_freq)
    for (mcap_file_path, _), df in zip(mcap_files, dataframes):
        data_df_to_mcap(df, mcap_file_path.replace(".mcap", "_processed.mcap"))

    print("Building spring/force dataset...")
    spring_windows, force_windows, spring_targets, force_targets = build_estimated_spring_dataset(
        dataframes,
        mcap_files,
        history_size=history_size,
        stride=stride,
        prediction=prediction,
        velocity_threshold=velocity_threshold,
        device=device,
    )

    # Normalize inputs and targets once. The shared trainer receives normalized tensors.
    spring_windows_norm, spring_input_mean, spring_input_std = normalize_tensor(spring_windows)
    spring_targets_norm, spring_output_mean, spring_output_std = normalize_tensor(spring_targets)
    force_windows_norm, force_input_base_mean, force_input_base_std = normalize_tensor(force_windows)
    force_targets_norm, force_output_mean, force_output_std = normalize_tensor(force_targets)

    # Package inputs/outputs for the shared trainer.
    # combined_inputs: [N, 2, History, Feature] where dim 1 slices are [spring_window, force_window].
    # combined_targets: [N, 1, 2] with channels [force, spring].
    combined_inputs = torch.stack([spring_windows_norm, force_windows_norm], dim=1)
    combined_targets = torch.cat([force_targets_norm, spring_targets_norm], dim=-1)

    # Create transformers with sizes derived from the data.
    spring_transformer = TorchTransformerModel(
        input_size=spring_windows.shape[-1],
        output_size=spring_targets.shape[-1],
        num_layers=2,
        history_size=history_size,
        num_heads=2,
        hidden_dim=32,
        device=device,
    )
    force_transformer = TorchTransformerModel(
        input_size=force_windows.shape[-1] + spring_targets.shape[-1],
        output_size=force_targets.shape[-1],
        num_layers=2,
        history_size=history_size,
        num_heads=2,
        hidden_dim=32,
        device=device,
    )

    training_model = SpringForceTrainingModel(
        spring_transformer=spring_transformer,
        force_transformer=force_transformer,
    ).to(device)

    # Assemble deployable model.
    deployable_model = SpringTransformerForceEstimator(
        spring_transformer=spring_transformer,
        force_transformer=force_transformer,
        input_mean=force_input_base_mean,
        input_std=force_input_base_std,
        spring_input_mean=spring_input_mean,
        spring_input_std=spring_input_std,
        velocity_idx=INPUT_COLS.index("measured_velocity_rad_per_sec_data"),
        velocity_threshold=velocity_threshold,
    ).to(device)

    # Output stats are per-channel: force uses force output stats, spring uses spring output stats.
    combined_output_mean = torch.cat([force_output_mean, spring_output_mean], dim=-1)
    combined_output_std = torch.cat([force_output_std, spring_output_std], dim=-1)

    wrapped_model = ScaledModelWrapper(
        deployable_model,
        force_input_base_mean,
        force_input_base_std,
        combined_output_mean,
        combined_output_std,
        frequency=inference_freq,
        history_size=history_size,
        stride=stride,
        prediction=prediction,
        input_columns=INPUT_COLS,
        output_columns=output_cols,
    )
    model_saver = ModelSaver(wrapped_model, OUTPUT_DIR)

    train(
        training_model,
        combined_inputs,
        combined_targets,
        model_saver=model_saver,
        latest_prefix="estimated_spring_transformer_",
        loss_fn=make_spring_force_loss(aux_weight),
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_ratio=train_ratio,
    )


if __name__ == "__main__":
    main()
