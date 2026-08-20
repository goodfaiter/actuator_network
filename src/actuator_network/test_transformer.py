import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch, process_inputs_time_series


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_03_30_0.mcap",  # finger, mixed 200Hz
        "/workspace/data/training_data/2026_08_20/rosbag2_2026_08_20-08_52_16_0.mcap",  # finger, mixed 200Hz
    ]

    file_path = "/workspace/data/output_data/best_latest.pt"
    model = torch.jit.load(file_path, map_location="cpu")

    data_freq = 200
    stride = int(model.stride.item())
    num_hist = int(model.history_size.item())
    prediction = bool(model.prediction_mode.item())
    input_cols = model.input_columns
    output_cols = model.output_columns

    for mcap_file_path in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=data_freq)
        # Remove duplicate timestamps by keeping the first occurrence
        data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
        process_dataframe(data_df_extrapolated)
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device="cpu")
        input_indices = [col_names.index(col) for col in input_cols]
        inputs = process_inputs_time_series(
            data_tensor[:, input_indices], history_size=num_hist, stride=stride, prediction=prediction
        )

        # Run all the samples and save to the dataframe
        predictions = torch.zeros((inputs.shape[0], len(output_cols)))
        for col in output_cols:
            data_df_extrapolated[col + "_predicted"] = 0.0

        with torch.no_grad():
            preds = model(inputs)
        predictions[:, :] = preds[:, 0, :]  # Take the first time step
        offset = (num_hist - 1) * stride
        for i, col in enumerate(output_cols):
            data_df_extrapolated.loc[
                data_df_extrapolated.index[offset : offset + predictions.shape[0]],
                col + "_predicted",
            ] = predictions[:, i].numpy()

        # Save the dataframe with predictions
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_predicted.mcap"))


if __name__ == "__main__":
    main()
