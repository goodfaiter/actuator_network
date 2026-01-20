from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe
from helpers.pandas_to_torch import pandas_to_torch, process_inputs, process_outputs, normalize_tensor
from helpers.wrapper import ScaledModelWrapper
from helpers.torch_model import TorchMlpModel
from helpers.trainer import train
import os

os.environ["WANDB_API_KEY"] = ""


def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    stride = 4  # Stride for history and future steps, note stride 4 means our final freq is 20Hz
    num_hist = 3  # Number of history steps
    input_cols = ["desired_position_rad_data", "measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["weight_data"]
    mcap_file_paths = [
        "/workspace/data/training_data/2026_01_17/rosbag2_2026_01_17_14_37_22_0_recovered.mcap",  # test data, delete later
    ]

    data_df = read_mcap_to_dataframe(mcap_file_paths[0])
    data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
    col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device="cpu")
    input_indices = [col_names.index(col) for col in input_cols]
    output_indices = [col_names.index(col) for col in output_cols]
    inputs = process_inputs(data_tensor[:, input_indices], stride=stride, num_hist=num_hist)
    outputs = process_outputs(data_tensor[:, output_indices], stride=stride)
    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(outputs)
    model = TorchMlpModel(input_size=inputs_normalized.shape[1], output_size=outputs_normalized.shape[1], hidden_layers=[32, 32])
    wrapped_model = ScaledModelWrapper(model, inputs_mean, inputs_std, outputs_mean, outputs_std)
    train(model, inputs_normalized, outputs_normalized, model_to_save=wrapped_model)

    # plot data for testing
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 8))
    # for column in data.columns:
    #     plt.plot(data.index, data[column], label=column)
    # plt.xlabel("Time")
    # plt.ylabel("Values")
    # plt.title("Extrapolated Data from MCAP")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
