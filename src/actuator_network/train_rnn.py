"""Train a GRU-based RNN to estimate tendon force from position and velocity."""

import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.hyperparameters import RnnConfig
from actuator_network.helpers.pandas_to_torch import apply_normalization, normalize_tensor
from actuator_network.helpers.torch_model import TorchRNNModel
from actuator_network.helpers.trainer import train_stateful
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper

OUTPUT_DIR = "/workspace/data/output_data/"
DEFAULT_WANDB_PROJECT = "actuator_network"


def train_rnn(
    config: RnnConfig,
    train_input_chunks: torch.Tensor,
    train_output_chunks: torch.Tensor,
    val_input_chunks: torch.Tensor,
    val_output_chunks: torch.Tensor,
    device: torch.device,
) -> None:
    """Train an RNN with the given configuration.

    Args:
        config: Hyperparameter configuration.
        train_input_chunks: Training input sequences.
        train_output_chunks: Training target sequences.
        val_input_chunks: Validation input sequences.
        val_output_chunks: Validation target sequences.
        device: Torch device.
    """
    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(train_input_chunks)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(train_output_chunks)
    val_inputs_normalized = apply_normalization(val_input_chunks, inputs_mean, inputs_std)
    val_outputs_normalized = apply_normalization(val_output_chunks, outputs_mean, outputs_std)

    # Targets are the full output chunks (per-timestep predictions).
    train_targets = outputs_normalized
    val_targets = val_outputs_normalized

    model = TorchRNNModel(
        input_size=inputs_normalized.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=train_targets.shape[-1],
        device=device,
        dropout=config.dropout,
    )
    wrapped_model = ScaledModelWrapper(
        model,
        inputs_mean,
        inputs_std,
        outputs_mean,
        outputs_std,
        frequency=config.data_freq,
        history_size=config.seq_length,
        stride=config.stride,
        seq_length=config.seq_length,
        prediction=config.prediction,
        input_columns=config.input_cols,
        output_columns=config.output_cols,
    )
    model_saver = ModelSaver(wrapped_model, OUTPUT_DIR)

    train_stateful(
        model,
        inputs_normalized,
        train_targets,
        val_inputs_normalized,
        val_targets,
        model_saver,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        chunk_batch_size=config.chunk_batch_size,
        max_grad_norm=config.max_grad_norm,
        latest_prefix="rnn_",
        val_fraction=config.val_fraction,
    )


def main():
    if wandb.run is None:
        wandb.init(project=DEFAULT_WANDB_PROJECT)

    config = RnnConfig.from_wandb_config(wandb.config)
    print(f"Using configuration: {config}")

    if not config.is_valid():
        print(f"Skipping invalid configuration: {config}")
        wandb.run.summary["skipped_invalid"] = True
        wandb.finish()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_19/rosbag2_2026_08_19-12_40_03_0.mcap",
    ]
    val_mcap_file_paths = [
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-11_58_32_0.mcap",
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_18_38_0.mcap",
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_34_43_0.mcap",
    ]

    train_input_chunks, train_output_chunks = load_mcap_files_parallel(
        mcap_file_paths,
        freq=config.data_freq,
        input_cols=config.input_cols,
        output_cols=config.output_cols,
        history_size=config.seq_length,
        stride=config.stride,
        prediction=config.prediction,
        rnn_mode=True,
    )
    train_input_chunks = train_input_chunks.to(device)
    train_output_chunks = train_output_chunks.to(device)

    val_input_chunks, val_output_chunks = load_mcap_files_parallel(
        val_mcap_file_paths,
        freq=config.data_freq,
        input_cols=config.input_cols,
        output_cols=config.output_cols,
        history_size=config.seq_length,
        stride=config.stride,
        prediction=config.prediction,
        rnn_mode=True,
    )
    val_input_chunks = val_input_chunks.to(device)
    val_output_chunks = val_output_chunks.to(device)

    train_rnn(
        config,
        train_input_chunks,
        train_output_chunks,
        val_input_chunks,
        val_output_chunks,
        device,
    )


if __name__ == "__main__":
    main()
