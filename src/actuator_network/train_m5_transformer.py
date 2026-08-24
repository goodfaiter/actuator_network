"""Train a Transformer and an M5 friction model jointly for tendon-force estimation."""

import json
import os

import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.pandas_to_torch import normalize_tensor
from actuator_network.helpers.torch_model import M5TransformerPhysicsModel, TorchTransformerModel
from actuator_network.helpers.trainer import data_generator, split_data
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper

M5_PARAMS_PATH = "/workspace/data/output_data/m5_friction_params.json"
MOTOR_GAIN = 4.2
OUTPUT_DIR = "/workspace/data/output_data/"


def load_m5_model(
    params_path: str,
    device: torch.device,
    trainable: bool = False,
) -> M5FrictionModel:
    """Load an M5 friction model from JSON parameters.

    Args:
        params_path: Path to the JSON file written by train_m5.py.
        device: Torch device to place the model on.
        trainable: If True, keep the model in train mode with gradients enabled.
            If False, freeze the model and use it as a fixed physics prior.

    Returns:
        M5FrictionModel initialized from JSON.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"M5 parameters not found at {params_path}. Run 'uv run train-m5' first to fit the M5 friction model."
        )

    with open(params_path) as f:
        params = json.load(f)

    model = M5FrictionModel().to(device)
    model.set_physical_parameters(params)
    if trainable:
        model.train()
        model.requires_grad_(True)
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def train_m5_transformer(
    model: M5TransformerPhysicsModel,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    model_saver: ModelSaver,
    latest_prefix: str = "",
    aux_weight: float = 0.1,
    max_grad_norm: float = 1.0,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 1024,
    train_ratio: float = 0.9,
) -> None:
    """Train the combined M5 + Transformer model with an auxiliary loss and gradient clipping.

    Args:
        model: The combined M5 + Transformer model to train.
        inputs: Normalized input tensor of shape (num_samples, history_size, input_dim).
        outputs: Normalized target tensor of shape (num_samples, 1, output_dim).
        model_saver: ModelSaver instance for checkpointing.
        latest_prefix: Prefix inserted before "best_"/"final_" in latest checkpoint names.
        aux_weight: Weight for the auxiliary MSE loss on the Transformer's tau_external prediction.
        max_grad_norm: Maximum gradient norm for clipping.
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        batch_size: Training batch size.
        train_ratio: Fraction of data to use for training.
    """
    wandb.init(project="actuator_network")
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "train_ratio": train_ratio,
            "aux_weight": aux_weight,
            "max_grad_norm": max_grad_norm,
        }
    )
    wandb.log({"Model": str(model)})

    inputs_train, outputs_train, inputs_val, outputs_val = split_data(inputs, outputs, train_ratio=train_ratio)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_final_loss = 0.0
        epoch_aux_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_outputs in data_generator(inputs_train, outputs_train, batch_size):
            optimizer.zero_grad()

            final_pred = model(batch_inputs)
            aux_pred = model.transformer(batch_inputs)

            final_loss = criterion(final_pred, batch_outputs)
            aux_loss = criterion(aux_pred, batch_outputs)
            loss = final_loss + aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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
            final_val_pred = model(inputs_val)
            aux_val_pred = model.transformer(inputs_val)

            val_final_loss = criterion(final_val_pred, outputs_val).item()
            val_aux_loss = criterion(aux_val_pred, outputs_val).item()
            val_loss = val_final_loss + aux_weight * val_aux_loss

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
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

        # Save every 100 epochs
        if (epoch + 1) % 100 == 0:
            model_saver.save_model(f"_epoch_{epoch + 1}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_saver.save_model("_best")
            model_saver.save_latest(f"best_{latest_prefix}")
            print(f"New best model! Val loss: {best_val_loss:.4f}")

    model_saver.save_model("_final")
    model_saver.save_latest(f"final_{latest_prefix}")
    wandb.finish()


def main():
    # Configuration
    data_freq = 200  # Desired frequency in Hz
    stride = 2  # Stride between future steps (2 for 100Hz prediction from 200Hz data)
    inference_freq = data_freq // stride  # Inference frequency in Hz
    prediction = False  # Whether we are doing prediction or estimation
    history_size = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_cols = ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["tendon_bota_force_newton_data"]
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_03_30_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_52_16_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_11_49_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_15_46_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_27_46_0.mcap",  # strong spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_31_31_0.mcap",  # strong spring, mixed 200Hz
    ]

    # Training knobs
    m5_trainable = True  # If False, M5 acts as a fixed physics prior.
    aux_weight = 0.1
    max_grad_norm = 1.0
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 1024
    train_ratio = 0.9

    print("Loading M5 friction model as initial guess...")
    m5_model = load_m5_model(M5_PARAMS_PATH, device, trainable=m5_trainable)
    print(f"  M5 trainable: {m5_trainable}")

    print("Loading and processing MCAP files...")
    all_inputs, all_outputs = load_mcap_files_parallel(
        mcap_file_paths,
        freq=data_freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=history_size,
        stride=stride,
        prediction=prediction,
    )
    all_inputs = all_inputs.to(device)
    all_outputs = all_outputs.to(device)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(all_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(all_outputs)

    delta_position_idx = input_cols.index("delta_position_rad_data")
    velocity_idx = input_cols.index("measured_velocity_rad_per_sec_data")

    transformer = TorchTransformerModel(
        input_size=inputs_normalized.shape[-1],
        output_size=outputs_normalized.shape[-1],
        num_layers=2,
        history_size=history_size,
        num_heads=4,
        hidden_dim=32,
        device=device,
    )

    combined_model = M5TransformerPhysicsModel(
        m5=m5_model,
        transformer=transformer,
        input_mean=inputs_mean,
        input_std=inputs_std,
        output_mean=outputs_mean,
        output_std=outputs_std,
        delta_position_idx=delta_position_idx,
        velocity_idx=velocity_idx,
        motor_gain=MOTOR_GAIN,
    )

    wrapped_model = ScaledModelWrapper(
        combined_model,
        inputs_mean,
        inputs_std,
        outputs_mean,
        outputs_std,
        frequency=inference_freq,
        history_size=history_size,
        stride=stride,
        prediction=prediction,
        input_columns=input_cols,
        output_columns=output_cols,
    )
    model_saver = ModelSaver(wrapped_model, OUTPUT_DIR)
    train_m5_transformer(
        combined_model,
        inputs_normalized,
        outputs_normalized,
        model_saver=model_saver,
        latest_prefix="m5_transformer_",
        aux_weight=aux_weight,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_ratio=train_ratio,
    )

    # Save the jointly fitted M5 parameters for inspection.
    joint_params = m5_model.named_physical_parameters()
    params_path = os.path.join(OUTPUT_DIR, "m5_joint_friction_params.json")
    with open(params_path, "w") as f:
        json.dump(joint_params, f, indent=2)
    print(f"Saved joint M5 parameters to {params_path}")


if __name__ == "__main__":
    main()
