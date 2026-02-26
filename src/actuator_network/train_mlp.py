import torch
from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from helpers.pandas_to_torch import pandas_to_torch, process_inputs, process_outputs, normalize_tensor
from helpers.pandas_to_mcap import data_df_to_mcap
from helpers.wrapper import ScaledModelWrapper, ModelSaver
from helpers.torch_model import TorchMlpModel
from helpers.trainer import train


def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    stride = 4  # Stride for history and future steps, note stride 4 means our final freq is 20Hz
    num_hist = 30  # Number of history steps
    prediction = False  # Whether we are doing prediction or estimation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_cols = ["desired_position_rad_data", "measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["load_newton_data"]
    mcap_file_paths = [
        ("/path/to/rosbag2.mcap", None), 
    ]

    all_inputs = torch.empty((0, len(input_cols) * num_hist), device=device)
    all_outputs = torch.empty((0, len(output_cols)), device=device)
    for mcap_file_path, spring_constant in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
        process_dataframe(data_df_extrapolated, spring_constant=spring_constant)
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_processed"))
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
        input_indices = [col_names.index(col) for col in input_cols]
        output_indices = [col_names.index(col) for col in output_cols]
        inputs = process_inputs(data_tensor[:, input_indices], stride=stride, num_hist=num_hist, prediction=prediction)
        outputs = process_outputs(data_tensor[:, output_indices], stride=stride, num_hist=num_hist, prediction=prediction)
        all_inputs = torch.cat((all_inputs, inputs), dim=0)
        all_outputs = torch.cat((all_outputs, outputs), dim=0)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(all_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(all_outputs)
    model = TorchMlpModel(input_size=inputs_normalized.shape[-1], output_size=outputs_normalized.shape[-1], hidden_layers=[256, 64, 16], device=device)
    wrapped_model = ScaledModelWrapper(model, inputs_mean, inputs_std, outputs_mean, outputs_std, frequency=freq, history_size=num_hist, stride=stride, prediction=prediction, input_columns=input_cols, output_columns=output_cols)
    model_saver = ModelSaver(wrapped_model, "/workspace/data/output_data/")
    train(model, inputs_normalized, outputs_normalized, model_saver=model_saver)


if __name__ == "__main__":
    main()
