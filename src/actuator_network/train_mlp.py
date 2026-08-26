import torch

from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.pandas_to_torch import apply_normalization, normalize_tensor
from actuator_network.helpers.torch_model import TorchMlpModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper


def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    stride = 4  # Stride for history steps, note stride 4 means our final freq is 20Hz
    num_hist = 30  # Number of history steps
    prediction = False  # Whether we are doing prediction or estimation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_cols = ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["tendon_bota_force_newton_data"]
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
        freq=freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=num_hist,
        stride=stride,
        prediction=prediction,
    )
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)

    val_inputs, val_outputs = load_mcap_files_parallel(
        val_mcap_file_paths,
        freq=freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=num_hist,
        stride=stride,
        prediction=prediction,
    )
    val_inputs = val_inputs.to(device)
    val_outputs = val_outputs.to(device)

    # Flatten windows for the MLP
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
        hidden_layers=[256, 64, 16],
        device=device,
    )
    wrapped_model = ScaledModelWrapper(
        model,
        inputs_mean,
        inputs_std,
        outputs_mean,
        outputs_std,
        frequency=freq,
        history_size=num_hist,
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
        latest_prefix="mlp_",
    )


if __name__ == "__main__":
    main()
