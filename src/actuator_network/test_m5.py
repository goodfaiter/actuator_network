"""Run M5 friction-model inference on test MCAPs and write predicted forces."""

import json

import torch

from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel
from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap

MODEL_PARAMS_PATH = "/workspace/data/output_data/m5_friction_params.json"


def load_m5_model(params_path: str, device: torch.device) -> M5FrictionModel:
    """Load an M5FrictionModel from a JSON parameter file."""
    with open(params_path) as f:
        params = json.load(f)

    model = M5FrictionModel().to(device)
    model.set_physical_parameters(params)
    model.eval()
    return model


def run_m5_inference(
    params_path: str,
    mcap_file_paths: list[str],
    data_freq: int,
) -> list[str]:
    """Run M5 inference on the given MCAPs and write prediction MCAPs.

    Returns:
        List of output file paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading M5 model...")
    model = load_m5_model(params_path, device=device)

    print("Loading and processing MCAP files in parallel...")
    dataframes = load_mcap_dataframes_parallel(mcap_file_paths, freq=data_freq)

    output_paths = []
    for mcap_file_path, df in zip(mcap_file_paths, dataframes):
        velocity = torch.tensor(df["measured_velocity_rad_per_sec_data"].to_numpy(), dtype=torch.float32, device=device)
        delta_position = torch.tensor(df["delta_position_rad_data"].to_numpy(), dtype=torch.float32, device=device)
        tau_external = torch.tensor(df["tendon_bota_force_newton_data"].to_numpy(), dtype=torch.float32, device=device)

        tau_motor = 4.2 * delta_position

        # Predict only on finite rows; leave NaNs as NaN in the output.
        valid_mask = torch.isfinite(velocity) & torch.isfinite(tau_motor) & torch.isfinite(tau_external)
        df["m5_newton_predicted"] = float("nan")

        if valid_mask.any():
            velocity_valid = velocity[valid_mask]
            tau_motor_valid = tau_motor[valid_mask]
            tau_external_valid = tau_external[valid_mask]

            with torch.no_grad():
                tau_friction = model(velocity_valid, tau_motor_valid, tau_external_valid)
            tau_external_predicted = tau_motor_valid - tau_friction

            df.loc[df.index[valid_mask.cpu().numpy()], "m5_newton_predicted"] = tau_external_predicted.cpu().numpy()

        output_path = mcap_file_path.replace(".mcap", "_m5_predicted.mcap")
        data_df_to_mcap(df, output_path)
        print(f"  wrote {output_path}")
        output_paths.append(output_path)

    return output_paths


def main():
    # Configuration
    data_freq = 200  # Hz, must match training
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_19/rosbag2_2026_08_19-12_40_03_0.mcap",
    ]

    run_m5_inference(MODEL_PARAMS_PATH, mcap_file_paths, data_freq)


if __name__ == "__main__":
    main()
