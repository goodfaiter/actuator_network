import torch
from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from helpers.pandas_to_torch import pandas_to_torch, process_inputs, process_inputs_time_series
from helpers.pandas_to_mcap import data_df_to_mcap


def main():
    mcap_file_paths = [
        # ("/workspace/data/training_data/2026_01_28/rosbag2_2026_01_28-14_00_33_0.mcap", None), # test
        # ("/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-14_59_00_0.mcap", None),
        ("/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-15_05_30_0.mcap", None),  # untrained, test
        # ("/workspace/data/training_data/2026_02_02/rosbag2_2026_02_02-16_50_51_0.mcap", None),
        ("/workspace/data/training_data/2026_02_02/rosbag2_2026_02_02-16_58_13_0.mcap", None), # untrained, test
        ("/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-16_51_17_0.mcap", None), # finger slow step test
        ("/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-17_05_21_0.mcap", None), # weak spring slow step tes
    ]

    file_path = "/workspace/data/output_data/final_latest.pt"
    model = torch.jit.load(file_path, map_location="cpu")

    data_freq = 80
    stride = int(model.stride.item())
    num_hist = int(model.history_size.item())
    seq_length = int(model.seq_length.item())
    prediction = bool(model.prediction_mode.item())
    input_cols = model.input_columns
    output_cols = model.output_columns

    model_type = model.model_type  # TorchTransformerModel, TorchRNNModel, or TorchMlpModel

    for mcap_file_path, spring_constant in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=data_freq)
        process_dataframe(data_df_extrapolated, spring_constant=spring_constant)
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device="cpu")
        input_indices = [col_names.index(col) for col in input_cols]
        if model_type == "TorchRNNModel":
            inputs = process_inputs_time_series(data_tensor[:, input_indices], sequence_length=1, prediction=prediction)
        elif model_type == "TorchTransformerModel":
            inputs = process_inputs_time_series(data_tensor[:, input_indices], history_size=num_hist, stride=stride, prediction=prediction)
        else:
            inputs = process_inputs(data_tensor[:, input_indices], stride=stride, num_hist=num_hist, prediction=prediction)

        # Run all the samples and save to the dataframe
        predictions = torch.zeros((inputs.shape[0], len(output_cols)))
        for col in output_cols:
            data_df_extrapolated[col + "_predicted"] = 0.0

        if model_type in ["TorchTransformerModel"]:
            with torch.no_grad():
                preds = model(inputs)
            predictions[:, :] = preds[:, 0, :]  # Take the first time step
            for i, col in enumerate(output_cols):
                data_df_extrapolated[col + "_predicted"].iloc[:] = predictions[:, i].numpy()

        if model_type in ["TorchMlpModel"]:
            with torch.no_grad():
                preds = model(inputs)
            predictions[:, :] = preds
            for i, col in enumerate(output_cols):
                data_df_extrapolated[col + "_predicted"].iloc[(num_hist + (0 if prediction else -1)) * stride :] = predictions[:, i].numpy()

        # for i in range (inputs.shape[0]):
        #     input_sample = inputs[i, ...].unsqueeze(0)  # Keep batch dimension
        #     with torch.no_grad():
        #         pred = model(input_sample)
        #     predictions[i, :] = pred

        # for i, col in enumerate(output_cols):
        #     data_df_extrapolated[col + "_predicted"].iloc[(num_hist + (0 if prediction else -1)) * stride:] = predictions[:, i].numpy()

        # Save the dataframe with predictions
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_predicted"))


if __name__ == "__main__":
    main()
