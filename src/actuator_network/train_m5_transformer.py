"""Train a Transformer that predicts tendon force, then refines it through the M5 physics model."""

import json
import os

import torch

from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.pandas_to_torch import normalize_tensor
from actuator_network.helpers.torch_model import M5TransformerPhysicsModel, TorchTransformerModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper

M5_PARAMS_PATH = "/workspace/data/output_data/m5_friction_params.json"
MOTOR_GAIN = 4.2


def load_m5_model(params_path: str, device: torch.device) -> M5FrictionModel:
    """Load a pre-trained M5 friction model from JSON parameters.

    Args:
        params_path: Path to the JSON file written by train_m5.py.
        device: Torch device to place the model on.

    Returns:
        Frozen M5FrictionModel in eval mode.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"M5 parameters not found at {params_path}. Run 'uv run train-m5' first to fit the M5 friction model."
        )

    with open(params_path) as f:
        params = json.load(f)

    model = M5FrictionModel().to(device)
    model.set_physical_parameters(params)
    model.eval()
    model.requires_grad_(False)
    return model


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
    ]

    print("Loading pre-trained M5 friction model...")
    m5_model = load_m5_model(M5_PARAMS_PATH, device)

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
    model_saver = ModelSaver(wrapped_model, "/workspace/data/output_data/")
    train(
        combined_model,
        inputs_normalized,
        outputs_normalized,
        model_saver=model_saver,
        latest_prefix="m5_transformer_",
    )


if __name__ == "__main__":
    main()
