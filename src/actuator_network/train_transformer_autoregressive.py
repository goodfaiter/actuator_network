"""Train an autoregressive Transformer with teacher forcing.

The model predicts tendon_bota_force_newton_data and uses the previous timestep's
force as an additional input channel (teacher forcing during training).
"""

import pandas as pd
import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel
from actuator_network.helpers.hyperparameters import TransformerAutoregressiveConfig
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import (
    apply_normalization,
    normalize_tensor,
    pandas_to_torch,
    process_inputs_time_series,
    process_outputs_time_series,
)
from actuator_network.helpers.torch_model import TorchTransformerModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper

OUTPUT_DIR = "/workspace/data/output_data/"
DEFAULT_WANDB_PROJECT = "actuator_network"


def build_autoregressive_dataset(
    dataframes: list[pd.DataFrame],
    input_cols: list[str],
    output_cols: list[str],
    history_size: int,
    stride: int,
    prediction: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build teacher-forced input/output tensors from processed DataFrames.

    The autoregressive channel (the last column in ``input_cols``) is shifted by
    one timestep, so the model predicts force at time ``t`` from history that ends
    with force at ``t-1``.
    """
    all_inputs = []
    all_outputs = []

    for df in dataframes:
        col_names, data_tensor = pandas_to_torch(df, device="cpu")
        input_indices = [col_names.index(col) for col in input_cols]
        output_indices = [col_names.index(col) for col in output_cols]

        features = data_tensor[:, input_indices]
        shifted_features = features.clone()
        # The last input column is the autoregressive force channel.
        shifted_features[1:, -1] = features[:-1, -1]
        shifted_features[0, -1] = 0.0

        inputs = process_inputs_time_series(
            shifted_features,
            history_size=history_size,
            stride=stride,
            prediction=prediction,
        )
        outputs = process_outputs_time_series(
            data_tensor[:, output_indices],
            stride=stride,
            history_size=history_size,
        )

        all_inputs.append(inputs)
        all_outputs.append(outputs)

    return torch.cat(all_inputs, dim=0).to(device), torch.cat(all_outputs, dim=0).to(device)


def train_transformer_autoregressive(
    config: TransformerAutoregressiveConfig,
    train_dataframes: list[pd.DataFrame],
    val_dataframes: list[pd.DataFrame],
    device: torch.device,
) -> None:
    """Train an autoregressive Transformer with the given configuration.

    Args:
        config: Hyperparameter configuration.
        train_dataframes: Processed training DataFrames.
        val_dataframes: Processed validation DataFrames.
        device: Torch device.
    """
    train_inputs, train_outputs = build_autoregressive_dataset(
        train_dataframes,
        input_cols=config.input_cols,
        output_cols=config.output_cols,
        history_size=config.history_size,
        stride=config.stride,
        prediction=config.prediction,
        device=device,
    )
    val_inputs, val_outputs = build_autoregressive_dataset(
        val_dataframes,
        input_cols=config.input_cols,
        output_cols=config.output_cols,
        history_size=config.history_size,
        stride=config.stride,
        prediction=config.prediction,
        device=device,
    )

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(train_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(train_outputs)
    val_inputs_normalized = apply_normalization(val_inputs, inputs_mean, inputs_std)
    val_outputs_normalized = apply_normalization(val_outputs, outputs_mean, outputs_std)

    model = TorchTransformerModel(
        input_size=inputs_normalized.shape[-1],
        output_size=outputs_normalized.shape[-1],
        num_layers=config.num_layers,
        history_size=config.history_size,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        device=device,
        dropout=config.dropout,
        activation=config.activation,
    )
    wrapped_model = ScaledModelWrapper(
        model,
        inputs_mean,
        inputs_std,
        outputs_mean,
        outputs_std,
        frequency=config.inference_freq,
        history_size=config.history_size,
        stride=config.stride,
        prediction=config.prediction,
        input_columns=config.input_cols,
        output_columns=config.output_cols,
    )
    model_saver = ModelSaver(wrapped_model, OUTPUT_DIR)
    train(
        model,
        inputs_normalized,
        outputs_normalized,
        val_inputs_normalized,
        val_outputs_normalized,
        model_saver=model_saver,
        latest_prefix="transformer_autoregressive_",
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        weight_decay=config.weight_decay,
        scheduler_type=config.scheduler_type,
        max_grad_norm=config.max_grad_norm,
    )


def main():
    if wandb.run is None:
        wandb.init(project=DEFAULT_WANDB_PROJECT)

    config = TransformerAutoregressiveConfig.from_wandb_config(wandb.config)
    print(f"Using configuration: {config}")

    if not config.is_valid():
        print(f"Skipping invalid configuration: {config}")
        wandb.run.summary["skipped_invalid"] = True
        wandb.finish()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_03_30_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_52_16_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_11_49_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_15_46_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_27_46_0.mcap",  # strong spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_31_31_0.mcap",  # strong spring, mixed 200Hz
    ]
    val_mcap_file_paths = [
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-11_58_32_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_18_38_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_34_43_0.mcap",  # strong spring, mixed 200Hz
    ]

    print("Loading and processing training MCAP files in parallel...")
    train_dataframes = load_mcap_dataframes_parallel(mcap_file_paths, freq=config.data_freq)
    for mcap_file_path, df in zip(mcap_file_paths, train_dataframes):
        data_df_to_mcap(df, mcap_file_path.replace(".mcap", "_processed.mcap"))

    print("Loading and processing validation MCAP files in parallel...")
    val_dataframes = load_mcap_dataframes_parallel(val_mcap_file_paths, freq=config.data_freq)
    for mcap_file_path, df in zip(val_mcap_file_paths, val_dataframes):
        data_df_to_mcap(df, mcap_file_path.replace(".mcap", "_processed.mcap"))

    train_transformer_autoregressive(
        config,
        train_dataframes,
        val_dataframes,
        device,
    )


if __name__ == "__main__":
    main()
