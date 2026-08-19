import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import (
    normalize_tensor,
    pandas_to_torch,
    process_inputs_time_series,
    process_outputs_time_series,
)
from actuator_network.helpers.torch_model import TorchTransformerModel
from actuator_network.helpers.trainer import train
from actuator_network.helpers.wrapper import ModelSaver, ScaledModelWrapper


def main():
    # Configuration
    data_freq = 80  # Desired frequency in Hz
    stride = 4  # Stride between future steps (4 for 20Hz prediction from 80Hz data)
    inference_freq = data_freq // stride  # Inference frequency in Hz
    prediction = False  # Whether we are doing prediction or estimation
    history_size = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_cols = ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["load_newton_data"]
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_19/rosbag2_2026_08_19-12_40_03_0.mcap",
    ]

    all_inputs = torch.empty((0, history_size, len(input_cols)), device=device)
    all_outputs = torch.empty((0, 1, len(output_cols)), device=device)
    for mcap_file_path in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=data_freq)
        data_df_extrapolated = data_df_extrapolated.groupby(
            data_df_extrapolated.index
        ).first()  # Remove duplicate timestamps by keeping the first occurrence
        process_dataframe(data_df_extrapolated)
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_processed"))
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
        input_indices = [col_names.index(col) for col in input_cols]
        output_indices = [col_names.index(col) for col in output_cols]
        inputs = process_inputs_time_series(
            data_tensor[:, input_indices], stride=stride, history_size=history_size, prediction=prediction
        )
        outputs = process_outputs_time_series(
            data_tensor[:, output_indices], stride=stride, history_size=history_size, prediction=prediction
        )
        all_inputs = torch.cat((all_inputs, inputs), dim=0)
        all_outputs = torch.cat((all_outputs, outputs), dim=0)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(all_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(all_outputs)
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
    train(model, inputs_normalized, outputs_normalized, model_saver=model_saver)


if __name__ == "__main__":
    main()
