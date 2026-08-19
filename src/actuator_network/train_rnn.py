import torch

import wandb
from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import normalize_tensor, pandas_to_torch
from actuator_network.helpers.rnn_pipeline import make_contiguous_chunks
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

    all_input_chunks = torch.empty((0, seq_length, len(input_cols)), device=device)
    all_output_chunks = torch.empty((0, seq_length, len(output_cols)), device=device)

    for mcap_file_path in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
        # Remove duplicate timestamps by keeping the first occurrence
        data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
        process_dataframe(data_df_extrapolated)
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_processed.mcap"))
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
        input_indices = [col_names.index(col) for col in input_cols]
        output_indices = [col_names.index(col) for col in output_cols]

        input_chunks = make_contiguous_chunks(data_tensor[:, input_indices], seq_length)
        output_chunks = make_contiguous_chunks(data_tensor[:, output_indices], seq_length)

        all_input_chunks = torch.cat((all_input_chunks, input_chunks), dim=0)
        all_output_chunks = torch.cat((all_output_chunks, output_chunks), dim=0)

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
