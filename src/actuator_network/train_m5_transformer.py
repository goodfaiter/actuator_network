"""Train a Transformer and an M5 friction model jointly for tendon-force estimation."""

import json
import os

import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.hyperparameters import M5TransformerConfig
from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.pandas_to_torch import apply_normalization, normalize_tensor
from actuator_network.helpers.torch_model import M5TransformerPhysicsModel, TorchTransformerModel
from actuator_network.helpers.trainer import data_generator

M5_PARAMS_PATH = "/workspace/data/output_data/m5_friction_params.json"
MOTOR_GAIN_DEFAULT = 4.2
OUTPUT_DIR = "/workspace/data/output_data/"
DEFAULT_WANDB_PROJECT = "actuator_network"


def load_m5_model(
    params_path: str,
    device: torch.device,
    trainable: bool = False,
    motor_gain_trainable: bool = True,
) -> M5FrictionModel:
    """Load an M5 friction model from JSON parameters.

    Args:
        params_path: Path to the JSON file written by train_m5.py.
        device: Torch device to place the model on.
        trainable: If True, enable gradients on the M5 friction parameters.
        motor_gain_trainable: If True, enable gradients on the motor gain P.

    Returns:
        M5FrictionModel initialized from JSON.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"M5 parameters not found at {params_path}. Run 'uv run train-m5' first to fit the M5 friction model."
        )

    with open(params_path) as f:
        params = json.load(f)

    motor_gain = params.get("motor_gain", MOTOR_GAIN_DEFAULT)
    model = M5FrictionModel(motor_gain=motor_gain, trainable_motor_gain=motor_gain_trainable).to(device)
    model.set_physical_parameters(params)
    model.set_friction_trainable(trainable)
    if trainable or motor_gain_trainable:
        model.train()
    else:
        model.eval()
    return model


def train_m5_transformer(
    config: M5TransformerConfig,
    model: M5TransformerPhysicsModel,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    val_inputs: torch.Tensor,
    val_outputs: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
) -> None:
    """Train the combined M5 + Transformer model with an auxiliary loss and gradient clipping.

    Nothing is checkpointed: the only persisted artifact for this pipeline is the
    jointly fitted M5 params JSON written by main() after training.

    Args:
        config: Hyperparameter configuration.
        model: The combined M5 + Transformer model to train.
        inputs: Normalized input tensor of shape (num_samples, history_size, input_dim).
        outputs: Normalized target tensor of shape (num_samples, 1, output_dim).
        val_inputs: Normalized validation input tensor.
        val_outputs: Normalized validation target tensor.
        input_mean: Training input mean of shape (1, input_dim), passed into the physics model.
        input_std: Training input standard deviation of shape (1, input_dim).
        output_mean: Training output mean of shape (1, output_dim).
        output_std: Training output standard deviation of shape (1, output_dim).
    """
    wandb.log({"Model": str(model)})

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    flat_input_mean = input_mean.view(-1)
    flat_input_std = input_std.view(-1)
    flat_output_mean = output_mean.view(-1)
    flat_output_std = output_std.view(-1)

    # Compute a single fixed random validation subset to save time.
    num_val_samples = val_inputs.shape[0]
    if config.val_fraction < 1.0 and num_val_samples > 0:
        subset_size = max(1, int(num_val_samples * config.val_fraction))
        val_indices = torch.randperm(num_val_samples)[:subset_size]
        val_inputs_subset = val_inputs[val_indices]
        val_outputs_subset = val_outputs[val_indices]
    else:
        val_inputs_subset = val_inputs
        val_outputs_subset = val_outputs

    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_final_loss = 0.0
        epoch_aux_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_outputs in data_generator(inputs, outputs, config.batch_size):
            optimizer.zero_grad()

            pred = model(
                batch_inputs, flat_input_mean, flat_input_std, flat_output_mean, flat_output_std
            )  # [Batch, 1, 4]

            # Channel 0 is the main predicted force; channel 3 is the Transformer's tau_external pred.
            final_loss = criterion(pred[:, :, 0:1], batch_outputs)
            aux_loss = criterion(pred[:, :, 3:4], batch_outputs)
            loss = final_loss + config.aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_final_loss += final_loss.item()
            epoch_aux_loss += aux_loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_final_loss = epoch_final_loss / max(num_batches, 1)
        avg_train_aux_loss = epoch_aux_loss / max(num_batches, 1)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(
                val_inputs_subset, flat_input_mean, flat_input_std, flat_output_mean, flat_output_std
            )  # [Batch, 1, 4]

            val_final_loss = criterion(val_pred[:, :, 0:1], val_outputs_subset).item()
            val_aux_loss = criterion(val_pred[:, :, 3:4], val_outputs_subset).item()
            val_loss = val_final_loss + config.aux_weight * val_aux_loss

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f} (final={avg_train_final_loss:.4f}, aux={avg_train_aux_loss:.4f}), "
            f"Val Loss: {val_loss:.4f} (final={val_final_loss:.4f}, aux={val_aux_loss:.4f})"
        )

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "train_final_loss": avg_train_final_loss,
                "train_aux_loss": avg_train_aux_loss,
                "val_loss": val_loss,
                "val_final_loss": val_final_loss,
                "val_aux_loss": val_aux_loss,
                "epoch": epoch + 1,
            }
        )

        # Track the best validation loss for logging only; nothing is checkpointed.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model! Val loss: {best_val_loss:.4f}")


def main():
    if wandb.run is None:
        wandb.init(project=DEFAULT_WANDB_PROJECT)

    config = M5TransformerConfig.from_wandb_config(wandb.config)
    print(f"Using configuration: {config}")

    if not config.is_valid():
        print(f"Skipping invalid configuration: {config}")
        wandb.run.summary["skipped_invalid"] = True
        wandb.finish()
        return

    wandb.config.update(
        {
            "m5_trainable": config.m5_trainable,
            "motor_gain_trainable": config.motor_gain_trainable,
        }
    )

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

    print("Loading M5 friction model as initial guess...")
    m5_model = load_m5_model(
        M5_PARAMS_PATH,
        device,
        trainable=config.m5_trainable,
        motor_gain_trainable=config.motor_gain_trainable,
    )
    print(f"  M5 friction params trainable: {config.m5_trainable}")
    print(f"  Motor gain trainable: {config.motor_gain_trainable}")

    print("Loading and processing MCAP files...")
    train_inputs, train_outputs = load_mcap_files_parallel(
        mcap_file_paths,
        freq=config.data_freq,
        input_cols=config.input_cols,
        output_cols=config.output_cols,
        history_size=config.history_size,
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
        history_size=config.history_size,
        stride=config.stride,
        prediction=config.prediction,
    )
    val_inputs = val_inputs.to(device)
    val_outputs = val_outputs.to(device)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(train_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(train_outputs)
    val_inputs_normalized = apply_normalization(val_inputs, inputs_mean, inputs_std)
    val_outputs_normalized = apply_normalization(val_outputs, outputs_mean, outputs_std)

    delta_position_idx = config.input_cols.index("delta_position_rad_data")
    velocity_idx = config.input_cols.index("measured_velocity_rad_per_sec_data")

    transformer = TorchTransformerModel(
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

    combined_model = M5TransformerPhysicsModel(
        m5=m5_model,
        transformer=transformer,
        delta_position_idx=delta_position_idx,
        velocity_idx=velocity_idx,
    )

    train_m5_transformer(
        config,
        combined_model,
        inputs_normalized,
        outputs_normalized,
        val_inputs_normalized,
        val_outputs_normalized,
        inputs_mean,
        inputs_std,
        outputs_mean,
        outputs_std,
    )

    # Save the jointly fitted M5 parameters for inspection.
    joint_params = m5_model.named_physical_parameters()
    params_path = os.path.join(OUTPUT_DIR, "m5_joint_friction_params.json")
    with open(params_path, "w") as f:
        json.dump(joint_params, f, indent=2)
    print(f"Saved joint M5 parameters to {params_path}")
    print(f"  motor_gain = {joint_params['motor_gain']:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()
