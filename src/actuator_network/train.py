from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe
from helpers.pandas_to_torch import pandas_to_torch, process_inputs, process_outputs, normalize_tensor

def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    stride = 4  # Stride for history and future steps, note stride 4 means our final freq is 20Hz
    num_hist = 3  # Number of history steps
    mcap_file_paths = [
        "/workspace/data/training_data/2026_01_17/rosbag2_2026_01_17_14_37_22_0_recovered.mcap", # test data, delete later
    ]

    data_df = read_mcap_to_dataframe(mcap_file_paths[0])
    data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
    col_indices, data_tensor = pandas_to_torch(data_df_extrapolated, device='cpu')
    inputs = process_inputs(data_tensor, stride=stride, num_hist=num_hist)
    outputs = process_outputs(data_tensor, stride=stride)
    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(outputs)

    print(data_df_extrapolated.head())

    #plot data for testing
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
