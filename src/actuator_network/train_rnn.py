import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.pandas_to_torch import apply_normalization, normalize_tensor
from actuator_network.helpers.torch_model import TorchRNNModel
from actuator_network.helpers.trainer import train_stateful
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper


def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    prediction = False  # Whether we are doing prediction or estimation
    seq_length = 512  # Sequence length for GRU
    stride = 1
    num_epochs = 50
    learning_rate = 0.001
    dropout = 0.1
    chunk_batch_size = 4
    max_grad_norm = 1.0
    val_fraction = 1.0
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

    train_input_chunks, train_output_chunks = load_mcap_files_parallel(
        mcap_file_paths,
        freq=freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=seq_length,
        stride=stride,
        prediction=prediction,
        rnn_mode=True,
    )
    train_input_chunks = train_input_chunks.to(device)
    train_output_chunks = train_output_chunks.to(device)

    val_input_chunks, val_output_chunks = load_mcap_files_parallel(
        val_mcap_file_paths,
        freq=freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=seq_length,
        stride=stride,
        prediction=prediction,
        rnn_mode=True,
    )
    val_input_chunks = val_input_chunks.to(device)
    val_output_chunks = val_output_chunks.to(device)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(train_input_chunks)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(train_output_chunks)
    val_inputs_normalized = apply_normalization(val_input_chunks, inputs_mean, inputs_std)
    val_outputs_normalized = apply_normalization(val_output_chunks, outputs_mean, outputs_std)

    # Targets are the full output chunks (per-timestep predictions)
    train_targets = outputs_normalized
    val_targets = val_outputs_normalized

    model = TorchRNNModel(
        input_size=inputs_normalized.shape[-1],
        hidden_size=64,
        num_layers=4,
        output_size=train_targets.shape[-1],
        device=device,
        dropout=dropout,
    )
    wrapped_model = ScaledModelWrapper(
        model,
        inputs_mean,
        inputs_std,
        outputs_mean,
        outputs_std,
        frequency=freq,
        history_size=seq_length,
        stride=stride,
        seq_length=seq_length,
        prediction=prediction,
        input_columns=input_cols,
        output_columns=output_cols,
    )
    model_saver = ModelSaver(wrapped_model, "/workspace/data/output_data/")

    wandb.init(project="actuator_network")
    train_stateful(
        model,
        inputs_normalized,
        train_targets,
        val_inputs_normalized,
        val_targets,
        model_saver,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        chunk_batch_size=chunk_batch_size,
        max_grad_norm=max_grad_norm,
        latest_prefix="rnn_",
        val_fraction=val_fraction,
    )


if __name__ == "__main__":
    main()
