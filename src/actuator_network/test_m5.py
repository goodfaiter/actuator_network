"""Run M5 friction-model inference on test MCAPs and write predicted forces."""

import json

import torch

from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel
from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch
from actuator_network.helpers.torch_model import PlainM5PhysicsModel
from actuator_network.helpers.wrapper import ScaledModelWrapper

MODEL_PARAMS_PATH = "/workspace/data/output_data/m5_friction_params.json"

INPUT_COLUMNS = ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
OUTPUT_COLUMNS = [
    "tendon_bota_force_newton_data",
    "tau_motor_newton_data",
    "tau_friction_newton_data",
    "tau_external_pred_newton_data",
]


def load_m5_model(params_path: str, device: torch.device) -> M5FrictionModel:
    """Load an M5FrictionModel from a JSON parameter file."""
    with open(params_path) as f:
        params = json.load(f)

    model = M5FrictionModel().to(device)
    model.set_physical_parameters(params)
    model.eval()
    return model


def _build_scripted_plain_m5(m5: M5FrictionModel, data_freq: int, device: torch.device):
    """Wrap a frozen M5 model so its JIT forward pass returns physics intermediates."""
    m5.eval()
    m5.requires_grad_(False)

    physics = PlainM5PhysicsModel(m5).to(device)
    physics.eval()

    input_mean = torch.zeros(len(INPUT_COLUMNS), device=device)
    input_std = torch.ones(len(INPUT_COLUMNS), device=device)
    output_mean = torch.zeros(len(OUTPUT_COLUMNS), device=device)
    output_std = torch.ones(len(OUTPUT_COLUMNS), device=device)

    wrapped = ScaledModelWrapper(
        physics,
        input_mean,
        input_std,
        output_mean,
        output_std,
        frequency=data_freq,
        history_size=1,
        stride=1,
        prediction=False,
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
    )
    wrapped.eval()
    return torch.jit.script(wrapped)


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
    m5 = load_m5_model(params_path, device=device)
    model = _build_scripted_plain_m5(m5, data_freq, device=device)

    print("Loading and processing MCAP files in parallel...")
    dataframes = load_mcap_dataframes_parallel(mcap_file_paths, freq=data_freq)

    output_paths = []
    for mcap_file_path, df in zip(mcap_file_paths, dataframes):
        col_names, data_tensor = pandas_to_torch(df, device=device)
        input_indices = [col_names.index(col) for col in model.input_columns]
        features = data_tensor[:, input_indices]

        output_cols = model.output_columns
        predictions = torch.full(
            (features.shape[0], len(output_cols)),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        # Predict only on finite rows; leave NaNs as NaN in the output.
        valid_mask = torch.isfinite(features).all(dim=1)
        if valid_mask.any():
            features_valid = features[valid_mask]
            with torch.no_grad():
                pred = model(features_valid)  # [N_valid, 1, 4]
            predictions[valid_mask] = pred[:, 0, :]

        for i, col in enumerate(output_cols):
            df[col + "_predicted"] = predictions[:, i].cpu().numpy()

        output_path = mcap_file_path.replace(".mcap", "_m5_predicted.mcap")
        data_df_to_mcap(df, output_path)
        print(f"  wrote {output_path}")
        output_paths.append(output_path)

    return output_paths


def main():
    # Configuration
    data_freq = 200  # Hz, must match training
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-11_58_32_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_18_38_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_34_43_0.mcap",  # strong spring, mixed 200Hz
    ]

    run_m5_inference(MODEL_PARAMS_PATH, mcap_file_paths, data_freq)


if __name__ == "__main__":
    main()
