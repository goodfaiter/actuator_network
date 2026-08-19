import os
import tempfile
from unittest.mock import patch

import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import normalize_tensor, pandas_to_torch
from actuator_network.helpers.rnn_pipeline import make_contiguous_chunks, run_stateful_inference
from actuator_network.helpers.torch_model import TorchRNNModel
from actuator_network.helpers.trainer import train_stateful
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper


def test_rnn_train_and_predict_smoke():
    mcap_path = "/workspace/tests/test.mcap"
    assert os.path.isfile(mcap_path), f"Test MCAP not found: {mcap_path}"

    freq = 80
    seq_length = 10
    train_ratio = 0.8
    num_epochs = 2
    input_cols = ["desired_position_rad_data", "measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["load_newton_data"]
    device = torch.device("cpu")

    data_df = read_mcap_to_dataframe(mcap_path)
    data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
    data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
    process_dataframe(data_df_extrapolated)
    col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
    input_indices = [col_names.index(col) for col in input_cols]
    output_indices = [col_names.index(col) for col in output_cols]

    input_data = data_tensor[:, input_indices]
    output_data = data_tensor[:, output_indices]

    input_chunks = make_contiguous_chunks(input_data, seq_length)
    output_chunks = make_contiguous_chunks(output_data, seq_length)
    assert input_chunks.shape[0] > 0, "Not enough data for the configured sequence length"

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(input_chunks)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(output_chunks)
    targets = outputs_normalized

    num_train = int(input_chunks.shape[0] * train_ratio)
    train_inputs = inputs_normalized[:num_train]
    train_targets = targets[:num_train]
    val_inputs = inputs_normalized[num_train:]
    val_targets = targets[num_train:]

    model = TorchRNNModel(
        input_size=len(input_cols),
        hidden_size=8,
        num_layers=1,
        output_size=len(output_cols),
        device=device,
        dropout=0.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        wrapped_model = ScaledModelWrapper(
            model,
            inputs_mean,
            inputs_std,
            outputs_mean,
            outputs_std,
            frequency=freq,
            history_size=seq_length,
            stride=1,
            seq_length=seq_length,
            prediction=False,
            input_columns=input_cols,
            output_columns=output_cols,
        )
        model_saver = ModelSaver(wrapped_model, tmpdir)

        with patch("actuator_network.helpers.trainer.wandb"):
            train_stateful(
                model,
                train_inputs,
                train_targets,
                val_inputs,
                val_targets,
                model_saver,
                num_epochs=num_epochs,
                learning_rate=0.01,
            )

        model_path = os.path.join(tmpdir, "final_latest.pt")
        assert os.path.isfile(model_path), "Final model was not saved"

        loaded_model = torch.jit.load(model_path, map_location="cpu")

        run_stateful_inference(
            loaded_model,
            input_data=input_data,
            output_cols=output_cols,
            seq_length=seq_length,
            data_df=data_df_extrapolated,
        )

        output_path = os.path.join(tmpdir, "test_predicted")
        data_df_to_mcap(data_df_extrapolated, output_path)

        expected_path = output_path + ".mcap"
        assert os.path.isfile(expected_path), "Predicted MCAP file was not created"
