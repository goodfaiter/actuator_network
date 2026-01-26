import torch
from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from helpers.pandas_to_torch import pandas_to_torch, process_inputs
from helpers.pandas_to_mcap import data_df_to_mcap


def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    stride = 4  # Stride for history and future steps, note stride 4 means our final freq is 20Hz
    num_hist = 3  # Number of history steps
    prediction = False  # Whether we are doing prediction or estimation
    input_cols = ["desired_position_rad_data", "measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["calculated_acceleration_meter_per_sec2_data", "load_newton_data"]
    mcap_file_paths = [
        ("/workspace/data/training_data/2026_01_21/rosbag2_2026_01_21-13_01_03_0.mcap", 1.26 / 4.81),  # spring 1 test
        # ("/workspace/data/training_data/2026_01_22/rosbag2_2026_01_22-16_20_19_0.mcap", None),  # Blocked data
        # ("/workspace/data/training_data/2026_01_22/rosbag2_2026_01_22-16_30_52_0.mcap", None), # Test with blocked
    ]

    file_path = "/workspace/data/output_data/latest.pt"
    model = torch.jit.load(file_path, map_location="cpu")

    for mcap_file_path, spring_constant in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
        process_dataframe(data_df_extrapolated, spring_constant=spring_constant)
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device="cpu")
        input_indices = [col_names.index(col) for col in input_cols]
        inputs = process_inputs(data_tensor[:, input_indices], stride=stride, num_hist=num_hist, prediction=prediction)

        # Run all the samples and save to the dataframe
        with torch.no_grad():
            predictions = model(inputs).cpu()
        # Add predictions to dataframe
        for i, col in enumerate(output_cols):
            data_df_extrapolated[col + "_predicted"] = 0.0
            data_df_extrapolated[col + "_predicted"].iloc[(num_hist + (0 if prediction else -1)) * stride:] = predictions[:, i].numpy()
        # Save the dataframe with predictions
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_predicted"))


if __name__ == "__main__":
    main()
