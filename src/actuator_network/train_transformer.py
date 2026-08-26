import torch

from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.pandas_to_torch import apply_normalization, normalize_tensor
from actuator_network.helpers.torch_model import TorchTransformerModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper


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
    val_mcap_file_paths = [
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-11_58_32_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_18_38_0.mcap",  # weak spring, mixed 200Hz
        "/workspace/data/training_data/2026_08_24/rosbag2_2026_08_24-13_34_43_0.mcap",  # strong spring, mixed 200Hz
    ]

    train_inputs, train_outputs = load_mcap_files_parallel(
        mcap_file_paths,
        freq=data_freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=history_size,
        stride=stride,
        prediction=prediction,
    )
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)

    val_inputs, val_outputs = load_mcap_files_parallel(
        val_mcap_file_paths,
        freq=data_freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=history_size,
        stride=stride,
        prediction=prediction,
    )
    val_inputs = val_inputs.to(device)
    val_outputs = val_outputs.to(device)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(train_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(train_outputs)
    val_inputs_normalized = apply_normalization(val_inputs, inputs_mean, inputs_std)
    val_outputs_normalized = apply_normalization(val_outputs, outputs_mean, outputs_std)

    model = TorchTransformerModel(
        input_size=inputs_normalized.shape[-1],
        output_size=outputs_normalized.shape[-1],
        num_layers=2,
        history_size=history_size,
        num_heads=4,
        hidden_dim=32,
        device=device,
    )
    wrapped_model = ScaledModelWrapper(
        model,
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
        model,
        inputs_normalized,
        outputs_normalized,
        val_inputs_normalized,
        val_outputs_normalized,
        model_saver=model_saver,
        latest_prefix="transformer_",
    )


if __name__ == "__main__":
    main()
