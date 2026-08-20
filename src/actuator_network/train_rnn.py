import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_files_parallel
from actuator_network.helpers.pandas_to_torch import normalize_tensor
from actuator_network.helpers.torch_model import TorchRNNModel
from actuator_network.helpers.trainer import train_stateful
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper


def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    prediction = False  # Whether we are doing prediction or estimation
    seq_length = 512  # Sequence length for GRU
    stride = 1
    train_ratio = 0.9
    num_epochs = 50
    learning_rate = 0.001
    dropout = 0.1
    chunk_batch_size = 4
    max_grad_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_cols = ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["tendon_bota_force_newton_data"]
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_19/rosbag2_2026_08_19-12_40_03_0.mcap",
    ]

    all_input_chunks, all_output_chunks = load_mcap_files_parallel(
        mcap_file_paths,
        freq=freq,
        input_cols=input_cols,
        output_cols=output_cols,
        history_size=seq_length,
        stride=stride,
        prediction=prediction,
        rnn_mode=True,
    )
    all_input_chunks = all_input_chunks.to(device)
    all_output_chunks = all_output_chunks.to(device)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(all_input_chunks)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(all_output_chunks)

    # Targets are the full output chunks (per-timestep predictions)
    targets_normalized = outputs_normalized

    # Contiguous train/val split by chunks
    num_train = int(inputs_normalized.shape[0] * train_ratio)
    train_inputs = inputs_normalized[:num_train]
    train_targets = targets_normalized[:num_train]
    val_inputs = inputs_normalized[num_train:]
    val_targets = targets_normalized[num_train:]

    model = TorchRNNModel(
        input_size=inputs_normalized.shape[-1],
        hidden_size=64,
        num_layers=4,
        output_size=targets_normalized.shape[-1],
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
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        model_saver,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        chunk_batch_size=chunk_batch_size,
        max_grad_norm=max_grad_norm,
    )


if __name__ == "__main__":
    main()
