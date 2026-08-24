"""Fit the M5 friction model to measured motor/external torques."""

import json
import os

import torch
import torch.nn.functional as functional

from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel
from actuator_network.helpers.m5_model import M5FrictionModel


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


def main():
    # Configuration
    data_freq = 200  # Hz
    num_epochs = 2000
    learning_rate = 0.01
    patience = 200
    trainable_motor_gain = False
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_03_30_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_52_16_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_11_49_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_15_46_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_27_46_0.mcap",  # strong spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_31_31_0.mcap",  # strong spring, mixed 200Hz
    ]
    output_dir = "/workspace/data/output_data/"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading and processing MCAP files in parallel...")
    dataframes = load_mcap_dataframes_parallel(mcap_file_paths, freq=data_freq)

    print("Preparing tensors...")
    velocity, delta_position, tau_external = prepare_tensors_from_dataframes(dataframes, device=device)
    print(f"  samples after cleaning: {velocity.shape[0]}")

    model = M5FrictionModel(motor_gain=4.2, trainable_motor_gain=trainable_motor_gain).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 4)

    best_loss = float("inf")
    epochs_without_improvement = 0

    print("Fitting M5 friction model...")
    for epoch in range(num_epochs):
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

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"  epoch {epoch:4d}: loss={current_loss:.6f}, best={best_loss:.6f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    params = model.named_physical_parameters()
    print("\nFitted parameters:")
    for name, value in params.items():
        print(f"  {name} = {value:.6f}")
    print(f"\nFinal MSE loss: {best_loss:.6f}")
    print(f"Final RMSE:     {best_loss**0.5:.6f}")

    # Save results
    model_path = os.path.join(output_dir, "m5_friction_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model state dict to {model_path}")

    params_path = os.path.join(output_dir, "m5_friction_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved parameters to {params_path}")


if __name__ == "__main__":
    main()
