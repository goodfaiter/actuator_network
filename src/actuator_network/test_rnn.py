import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch
from actuator_network.helpers.rnn_pipeline import run_stateful_inference


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_08_19/rosbag2_2026_08_19-12_40_03_0.mcap",
    ]

    file_path = "/workspace/data/output_data/best_latest.pt"
    model = torch.jit.load(file_path, map_location="cpu")

    data_freq = 80
    num_hist = int(model.history_size.item())
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

        run_stateful_inference(
            model,
            input_data=data_tensor[:, input_indices],
            output_cols=output_cols,
            seq_length=num_hist,
            data_df=data_df_extrapolated,
        )

        # Save the dataframe with predictions
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_predicted.mcap"))


if __name__ == "__main__":
    main()
