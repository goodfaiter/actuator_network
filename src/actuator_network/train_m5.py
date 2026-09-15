"""Fit the M5 friction model to measured motor/external torques."""

import json
import os

import torch
import torch.nn.functional as functional

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel
from actuator_network.helpers.hyperparameters import M5FrictionConfig
from actuator_network.helpers.m5_model import M5FrictionModel

OUTPUT_DIR = "/workspace/data/output_data/"
DEFAULT_WANDB_PROJECT = "actuator_network"
MOTOR_GAIN_DEFAULT = 4.2


def prepare_tensors_from_dataframes(
    dataframes: list,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate processed DataFrames and extract raw model inputs."""
    import pandas as pd

    df = pd.concat(dataframes, ignore_index=True)

    velocity = torch.tensor(df["measured_velocity_rad_per_sec_data"].to_numpy(), dtype=torch.float32, device=device)
    delta_position = torch.tensor(df["delta_position_rad_data"].to_numpy(), dtype=torch.float32, device=device)
    tau_external = torch.tensor(df["tendon_bota_force_newton_data"].to_numpy(), dtype=torch.float32, device=device)

    # Drop NaN/Inf rows
    valid_mask = torch.isfinite(velocity) & torch.isfinite(delta_position) & torch.isfinite(tau_external)
    velocity = velocity[valid_mask]
    delta_position = delta_position[valid_mask]
    tau_external = tau_external[valid_mask]

    return velocity, delta_position, tau_external


def train_m5_friction(
    config: M5FrictionConfig,
    dataframes: list,
    device: torch.device,
) -> M5FrictionModel:
    """Fit the M5 friction model with the given configuration.

    Args:
        config: Hyperparameter configuration.
        dataframes: Processed training DataFrames.
        device: Torch device.

    Returns:
        The fitted ``M5FrictionModel``.
    """
    velocity, delta_position, tau_external = prepare_tensors_from_dataframes(dataframes, device=device)
    print(f"  samples after cleaning: {velocity.shape[0]}")

    model = M5FrictionModel(
        motor_gain=MOTOR_GAIN_DEFAULT,
        trainable_motor_gain=config.trainable_motor_gain,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.patience // 4,
    )

    best_loss = float("inf")
    epochs_without_improvement = 0

    print("Fitting M5 friction model...")
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        tau_motor = model.compute_tau_motor(delta_position)
        target = tau_motor - tau_external
        prediction = model(velocity, tau_motor, tau_external)
        loss = functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 100 == 0 or epoch == config.num_epochs - 1:
            print(f"  epoch {epoch:4d}: loss={current_loss:.6f}, best={best_loss:.6f}")
            wandb.log({"epoch": epoch, "train_loss": current_loss, "best_loss": best_loss})

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {config.patience} epochs).")
            break

    params = model.named_physical_parameters()
    print("\nFitted parameters:")
    for name, value in params.items():
        print(f"  {name} = {value:.6f}")
    print(f"\nFinal MSE loss: {best_loss:.6f}")
    print(f"Final RMSE:     {best_loss**0.5:.6f}")

    wandb.log(
        {
            "final_mse_loss": best_loss,
            "final_rmse": best_loss**0.5,
            **{f"param/{k}": v for k, v in params.items()},
        }
    )

    return model


def main():
    if wandb.run is None:
        wandb.init(project=DEFAULT_WANDB_PROJECT)

    config = M5FrictionConfig.from_wandb_config(wandb.config)
    print(f"Using configuration: {config}")

    if not config.is_valid():
        print(f"Skipping invalid configuration: {config}")
        wandb.run.summary["skipped_invalid"] = True
        wandb.finish()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_03_30_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_52_16_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_11_49_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_15_46_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_27_46_0.mcap",  # strong spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_31_31_0.mcap",  # strong spring, mixed 200Hz
    ]

    print("Loading and processing MCAP files in parallel...")
    dataframes = load_mcap_dataframes_parallel(mcap_file_paths, freq=config.data_freq)

    print("Preparing tensors...")
    model = train_m5_friction(config, dataframes, device=device)

    # Save results
    model_path = os.path.join(OUTPUT_DIR, "m5_friction_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model state dict to {model_path}")

    params_path = os.path.join(OUTPUT_DIR, "m5_friction_params.json")
    with open(params_path, "w") as f:
        json.dump(model.named_physical_parameters(), f, indent=2)
    print(f"Saved parameters to {params_path}")


if __name__ == "__main__":
    main()
