"""Train a spring-class estimator + force estimator transformer pair jointly.

The spring transformer is trained to estimate a continuous spring coefficient
(0.0 = weak, 0.5 = finger, 1.0 = strong) from a frozen input buffer of
[delta_position, velocity]. The buffer is updated only when |velocity| > 0.1;
otherwise the previous buffer is reused. The force transformer receives
[delta_position, velocity, estimated_spring] and estimates
``tendon_bota_force_newton_data``. Both transformers are trained end-to-end
with a combined loss: force MSE + auxiliary spring MSE.

Hyperparameters are read from ``wandb.config`` so this script can be used as the
program for a W&B sweep agent. When run manually, it falls back to the defaults.
"""

import pandas as pd
import torch
import torch.nn as nn

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel_cached
from actuator_network.helpers.hyperparameters import EstimatedSpringTransformerConfig
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import (
    apply_normalization,
    normalize_tensor,
    pandas_to_torch,
)
from actuator_network.helpers.torch_model import (
    SpringForceTrainingModel,
    SpringTransformerForceEstimator,
    TorchTransformerModel,
)
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper

OUTPUT_DIR = "/workspace/data/output_data/"
DEFAULT_WANDB_PROJECT = "actuator_network"

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


def _build_aligned_windows(
    data: torch.Tensor,
    spring_history_size: int,
    force_history_size: int,
    spring_stride: int,
    force_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build spring and force windows aligned to the same end timestep.

    Both windows end at the same raw index, but each window samples backward at
    its own stride. This is necessary because the two transformers may use
    different history lengths and different strides.

    Args:
        data: Input tensor of shape ``(batch_size, feature_dim)``.
        spring_history_size: Length of the spring transformer's input window.
        force_history_size: Length of the force transformer's input window.
        spring_stride: Stride between spring history samples.
        force_stride: Stride between force history samples.

    Returns:
        Tuple of ``(spring_windows, force_windows)`` with shapes
        ``(num_sequences, spring_history_size, feature_dim)`` and
        ``(num_sequences, force_history_size, feature_dim)``.
    """
    batch_size, feature_dim = data.shape
    max_start = max((spring_history_size - 1) * spring_stride, (force_history_size - 1) * force_stride)
    num_sequences = batch_size - max_start
    if num_sequences <= 0:
        empty_shape_spring = (0, spring_history_size, feature_dim)
        empty_shape_force = (0, force_history_size, feature_dim)
        return torch.empty(empty_shape_spring, device=data.device), torch.empty(empty_shape_force, device=data.device)

    end_indices = torch.arange(num_sequences, device=data.device) + max_start

    spring_offsets = (
        torch.arange(spring_history_size, device=data.device) * spring_stride
        - (spring_history_size - 1) * spring_stride
    )
    spring_indices = end_indices.unsqueeze(1) + spring_offsets.unsqueeze(0)

    force_offsets = (
        torch.arange(force_history_size, device=data.device) * force_stride - (force_history_size - 1) * force_stride
    )
    force_indices = end_indices.unsqueeze(1) + force_offsets.unsqueeze(0)

    return data[spring_indices], data[force_indices]


def build_estimated_spring_dataset(
    dataframes: list[pd.DataFrame],
    file_labels: list[tuple[str, float]],
    spring_history_size: int,
    history_size: int,
    spring_stride: int,
    force_stride: int,
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
        spring_history_size: Length of the spring transformer's input window.
        history_size: Length of the force transformer's input window.
        spring_stride: Stride between spring history samples.
        force_stride: Stride between force history samples.
        prediction: Whether to shift outputs for prediction mode (currently unused).
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
        spring_windows, force_windows = _build_aligned_windows(
            features,
            spring_history_size=spring_history_size,
            force_history_size=history_size,
            spring_stride=spring_stride,
            force_stride=force_stride,
        )
        spring_windows = _build_frozen_spring_windows(
            spring_windows,
            velocity_idx=velocity_idx,
            velocity_threshold=velocity_threshold,
        )

        # Targets correspond to the shared end timestep of the aligned windows.
        max_start = max((spring_history_size - 1) * spring_stride, (history_size - 1) * force_stride)
        num_sequences = data_tensor.size(0) - max_start
        if num_sequences > 0:
            target_index = torch.arange(num_sequences, device=data_tensor.device) + max_start
            spring_targets = data_tensor[target_index, spring_idx].unsqueeze(1).unsqueeze(1)
            force_targets = data_tensor[target_index, output_idx].unsqueeze(1).unsqueeze(1)
        else:
            spring_targets = torch.empty((0, 1, 1), device=data_tensor.device)
            force_targets = torch.empty((0, 1, 1), device=data_tensor.device)

        all_spring_windows.append(spring_windows)
        all_force_windows.append(force_windows)
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


def make_input_noise_transform(noise_std: float):
    """Return a transform that adds Gaussian noise to normalized inputs.

    The transform accepts either a single tensor or a tuple of tensors and
    returns the same structure. Noise is sampled independently for every call,
    so the model sees different perturbations each epoch.

    Args:
        noise_std: Standard deviation of the additive Gaussian noise.

    Returns:
        Callable that adds noise to tensor(s).
    """

    def transform(inputs):
        if isinstance(inputs, tuple):
            return tuple(inp + torch.randn_like(inp) * noise_std for inp in inputs)
        return inputs + torch.randn_like(inputs) * noise_std

    return transform


def train_estimated_spring_transformer(
    config: EstimatedSpringTransformerConfig,
    train_dataframes: list[pd.DataFrame],
    val_dataframes: list[pd.DataFrame],
    mcap_files: list[tuple[str, float]],
    val_mcap_files: list[tuple[str, float]],
    device: torch.device,
    latest_prefix: str,
) -> None:
    """Train an estimated-spring transformer with the given configuration.

    Args:
        config: Hyperparameter configuration.
        train_dataframes: Processed training DataFrames.
        val_dataframes: Processed validation DataFrames.
        mcap_files: Training MCAP paths with spring labels.
        val_mcap_files: Validation MCAP paths with spring labels.
        device: Torch device.
        latest_prefix: Prefix for the latest checkpoint file name.
    """
    inference_freq = config.data_freq // config.force_stride
    output_cols = [OUTPUT_COL, SPRING_COL]

    print(
        "Building spring/force training dataset ("
        f"spring_history_size={config.spring_history_size}, spring_stride={config.spring_stride}, "
        f"force_history_size={config.force_history_size}, force_stride={config.force_stride})..."
    )
    train_spring_windows, train_force_windows, train_spring_targets, train_force_targets = (
        build_estimated_spring_dataset(
            train_dataframes,
            mcap_files,
            spring_history_size=config.spring_history_size,
            history_size=config.force_history_size,
            spring_stride=config.spring_stride,
            force_stride=config.force_stride,
            prediction=config.prediction,
            velocity_threshold=config.velocity_threshold,
            device=device,
        )
    )

    print("Building spring/force validation dataset...")
    val_spring_windows, val_force_windows, val_spring_targets, val_force_targets = build_estimated_spring_dataset(
        val_dataframes,
        val_mcap_files,
        spring_history_size=config.spring_history_size,
        history_size=config.force_history_size,
        spring_stride=config.spring_stride,
        force_stride=config.force_stride,
        prediction=config.prediction,
        velocity_threshold=config.velocity_threshold,
        device=device,
    )

    # Normalize inputs and targets using training statistics only.
    spring_windows_norm, spring_input_mean, spring_input_std = normalize_tensor(train_spring_windows)
    spring_targets_norm, spring_output_mean, spring_output_std = normalize_tensor(train_spring_targets)
    force_windows_norm, force_input_base_mean, force_input_base_std = normalize_tensor(train_force_windows)
    force_targets_norm, force_output_mean, force_output_std = normalize_tensor(train_force_targets)

    val_spring_windows_norm = apply_normalization(val_spring_windows, spring_input_mean, spring_input_std)
    val_spring_targets_norm = apply_normalization(val_spring_targets, spring_output_mean, spring_output_std)
    val_force_windows_norm = apply_normalization(val_force_windows, force_input_base_mean, force_input_base_std)
    val_force_targets_norm = apply_normalization(val_force_targets, force_output_mean, force_output_std)

    # Package inputs/outputs for the shared trainer.
    # combined_inputs: tuple of (spring_windows, force_windows) because the two
    # transformers may use different history lengths.
    # combined_targets: [N, 1, 2] with channels [force, spring].
    combined_inputs = (spring_windows_norm, force_windows_norm)
    combined_targets = torch.cat([force_targets_norm, spring_targets_norm], dim=-1)
    val_combined_inputs = (val_spring_windows_norm, val_force_windows_norm)
    val_combined_targets = torch.cat([val_force_targets_norm, val_spring_targets_norm], dim=-1)

    # Create transformers with sizes derived from the data.
    spring_transformer = TorchTransformerModel(
        input_size=train_spring_windows.shape[-1],
        output_size=train_spring_targets.shape[-1],
        num_layers=config.spring_num_layers,
        history_size=config.spring_history_size,
        num_heads=config.spring_num_heads,
        hidden_dim=config.spring_hidden_dim,
        device=device,
        dropout=config.spring_dropout,
        activation=config.spring_activation,
    )
    force_transformer = TorchTransformerModel(
        input_size=train_force_windows.shape[-1] + train_spring_targets.shape[-1],
        output_size=train_force_targets.shape[-1],
        num_layers=config.force_num_layers,
        history_size=config.force_history_size,
        num_heads=config.force_num_heads,
        hidden_dim=config.force_hidden_dim,
        device=device,
        dropout=config.force_dropout,
        activation=config.force_activation,
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
        velocity_threshold=config.velocity_threshold,
        spring_alpha=config.spring_alpha,
        spring_stride=config.spring_stride,
        force_stride=config.force_stride,
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
        history_size=config.force_history_size,
        stride=config.force_stride,
        spring_stride=config.spring_stride,
        spring_history_size=config.spring_history_size,
        prediction=config.prediction,
        input_columns=INPUT_COLS,
        output_columns=output_cols,
    )
    model_saver = ModelSaver(wrapped_model, OUTPUT_DIR)

    train(
        training_model,
        combined_inputs,
        combined_targets,
        val_combined_inputs,
        val_combined_targets,
        model_saver=model_saver,
        latest_prefix=latest_prefix,
        loss_fn=make_spring_force_loss(config.aux_weight),
        val_loss_fn=make_spring_force_loss(0.0),
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        weight_decay=config.weight_decay,
        scheduler_type=config.scheduler_type,
        max_grad_norm=config.max_grad_norm,
        input_transform=make_input_noise_transform(config.input_noise_std),
        accumulation_steps=config.accumulation_steps,
    )


def main():
    # When launched by `wandb agent`, this init picks up the sweep config. For a
    # manual run we explicitly set the project.
    if wandb.run is None:
        wandb.init(project=DEFAULT_WANDB_PROJECT)

    config = EstimatedSpringTransformerConfig.from_wandb_config(wandb.config)
    print(f"Using configuration: {config}")

    if not config.is_valid():
        print(f"Skipping invalid configuration: {config}")
        wandb.run.summary["skipped_invalid"] = True
        wandb.finish()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    val_mcap_files: list[tuple[str, float]] = [
        # finger, mixed 200Hz
        ("/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-11_58_32_0.mcap", 0.5),
        # weak spring, mixed 200Hz
        ("/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_18_38_0.mcap", 0.0),
        # strong spring, mixed 200Hz
        ("/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_34_43_0.mcap", 1.0),
    ]

    print("Loading and processing training MCAP files...")
    train_dataframes = load_mcap_dataframes_parallel_cached([path for path, _ in mcap_files], freq=config.data_freq)
    for (mcap_file_path, _), df in zip(mcap_files, train_dataframes):
        data_df_to_mcap(df, mcap_file_path.replace(".mcap", "_processed.mcap"))

    print("Loading and processing validation MCAP files...")
    val_dataframes = load_mcap_dataframes_parallel_cached([path for path, _ in val_mcap_files], freq=config.data_freq)
    for (mcap_file_path, _), df in zip(val_mcap_files, val_dataframes):
        data_df_to_mcap(df, mcap_file_path.replace(".mcap", "_processed.mcap"))

    latest_prefix = (
        f"estimated_spring_transformer_sweep_{wandb.run.id}_"
        if wandb.run.sweep_id is not None
        else "estimated_spring_transformer_"
    )

    print("Running training...")
    train_estimated_spring_transformer(
        config,
        train_dataframes,
        val_dataframes,
        mcap_files,
        val_mcap_files,
        device,
        latest_prefix=latest_prefix,
    )


if __name__ == "__main__":
    main()
