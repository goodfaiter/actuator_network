import torch
from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from helpers.pandas_to_torch import pandas_to_torch, process_inputs_for_rnn, normalize_tensor, process_outputs_for_rnn
from helpers.pandas_to_mcap import data_df_to_mcap
from helpers.wrapper import ScaledModelWrapper, ModelSaver
from helpers.torch_model import TorchRNNModel
from helpers.trainer import train
import os

os.environ["WANDB_API_KEY"] = ""


def main():
    # Configuration
    freq = 80  # Desired frequency in Hz
    prediction = False  # Whether we are doing prediction or estimation
    seq_length = 50 # Sequence length for RNN
    # input_cols = ["desired_position_rad_data", "measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
    input_cols = ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["calculated_acceleration_meter_per_sec2_data", "load_newton_data"]
    mcap_file_paths = [
        ("/workspace/data/training_data/2026_01_20/rosbag2_2026_01_20-18_28_50_0.mcap", 1.26 / 4.81),  # All spring 1 data, spring constant kg/rad
        # ("/workspace/data/training_data/2026_01_21/rosbag2_2026_01_21-12_55_29_0.mcap", 1.26 / 4.81),  # All spring 1 data
        # ("/workspace/data/training_data/2026_01_21/rosbag2_2026_01_21-12_58_39_0.mcap", 1.26 / 4.81),  # All spring 1 data
        # ("/workspace/data/training_data/2026_01_21/rosbag2_2026_01_21-13_00_05_0.mcap", 1.26 / 4.81),  # All spring 1 data
        # "/workspace/data/training_data/2026_01_21/rosbag2_2026_01_21-13_01_03_0.mcap", # Test with spring
        # ("/workspace/data/training_data/2026_01_22/rosbag2_2026_01_22-16_06_21_0.mcap", None),  # Blocked data
        # ("/workspace/data/training_data/2026_01_22/rosbag2_2026_01_22-16_20_19_0.mcap", None),  # Blocked data
        # "/workspace/data/training_data/2026_01_22/rosbag2_2026_01_22-16_30_52_0.mcap", # Test with blocked
    ]

    all_inputs = torch.empty((0, seq_length, len(input_cols)))
    all_outputs = torch.empty((0, 1, len(output_cols)))
    for mcap_file_path, spring_constant in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=freq)
        process_dataframe(data_df_extrapolated, spring_constant=spring_constant)
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_processed"))
        # data_df_extrapolated.to_csv(mcap_file_path.replace(".mcap", "_processed.csv"))
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device="cpu")
        input_indices = [col_names.index(col) for col in input_cols]
        output_indices = [col_names.index(col) for col in output_cols]
        inputs = process_inputs_for_rnn(data_tensor[:, input_indices], sequence_length=seq_length, prediction=prediction)
        outputs = process_outputs_for_rnn(data_tensor[:, output_indices], sequence_length=seq_length, prediction=prediction)
        all_inputs = torch.cat((all_inputs, inputs), dim=0)
        all_outputs = torch.cat((all_outputs, outputs), dim=0)

    inputs_normalized, inputs_mean, inputs_std = normalize_tensor(all_inputs)
    outputs_normalized, outputs_mean, outputs_std = normalize_tensor(all_outputs)
    model = TorchRNNModel(input_size=inputs_normalized.shape[-1], hidden_size=32, num_layers=2, output_size=outputs_normalized.shape[-1], device="cpu")
    wrapped_model = ScaledModelWrapper(model, inputs_mean, inputs_std, outputs_mean, outputs_std, frequency=freq, history_size=0, stride=0, prediction=prediction, input_columns=input_cols, output_columns=output_cols)
    model_saver = ModelSaver(wrapped_model, "/workspace/data/output_data/")
    train(model, inputs_normalized, outputs_normalized, model_saver=model_saver)


if __name__ == "__main__":
    main()
