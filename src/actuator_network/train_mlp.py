"""Train an MLP to estimate tendon force from position and velocity history."""

import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.hyperparameters import MlpConfig
from actuator_network.helpers.pandas_to_torch import apply_normalization, normalize_tensor
from actuator_network.helpers.torch_model import TorchMlpModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper

OUTPUT_DIR = "/workspace/data/output_data/"
DEFAULT_WANDB_PROJECT = "actuator_network"


def train_mlp(
    config: MlpConfig,
    train_inputs: torch.Tensor,
    train_outputs: torch.Tensor,
    val_inputs: torch.Tensor,
    val_outputs: torch.Tensor,
    device: torch.device,
) -> None:
    """Train an MLP with the given configuration.

    Args:
        config: Hyperparameter configuration.
        train_inputs: Training input windows.
        train_outputs: Training target values.
        val_inputs: Validation input windows.
        val_outputs: Validation target values.
        device: Torch device.
    """

    # Flatten windows for the MLP.
    train_inputs = train_inputs.view(train_inputs.shape[0], -1)
    val_inputs = val_inputs.view(val_inputs.shape[0], -1)
    train_outputs = train_outputs.squeeze(1)
    val_outputs = val_outputs.squeeze(1)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(train_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(train_outputs)
    val_inputs_normalized = apply_normalization(val_inputs, inputs_mean, inputs_std)
    val_outputs_normalized = apply_normalization(val_outputs, outputs_mean, outputs_std)

    model = TorchMlpModel(
        input_size=inputs_normalized.shape[-1],
        output_size=outputs_normalized.shape[-1],
        hidden_layers=config.hidden_layers,
        device=device,
    )
    wrapped_model = ScaledModelWrapper(
        model,
        inputs_mean,
        inputs_std,
        outputs_mean,
        outputs_std,
        frequency=config.data_freq,
        history_size=config.num_hist,
        stride=config.stride,
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
        latest_prefix="mlp_",
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

    config = MlpConfig.from_wandb_config(wandb.config)
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

    train_inputs, train_outputs = load_mcap_files_parallel(
        mcap_file_paths,
        freq=config.data_freq,
        input_cols=config.input_cols,
        output_cols=config.output_cols,
        history_size=config.num_hist,
        stride=config.stride,
        prediction=config.prediction,
    )
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)

    val_inputs, val_outputs = load_mcap_files_parallel(
        val_mcap_file_paths,
        freq=config.data_freq,
        input_cols=config.input_cols,
        output_cols=config.output_cols,
        history_size=config.num_hist,
        stride=config.stride,
        prediction=config.prediction,
    )
    val_inputs = val_inputs.to(device)
    val_outputs = val_outputs.to(device)

    train_mlp(
        config,
        train_inputs,
        train_outputs,
        val_inputs,
        val_outputs,
        device,
    )


if __name__ == "__main__":
    main()
