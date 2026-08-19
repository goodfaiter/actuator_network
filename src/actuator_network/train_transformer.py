import torch
from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from helpers.pandas_to_torch import pandas_to_torch, process_inputs_time_series, normalize_tensor, process_outputs_time_series
from helpers.pandas_to_mcap import data_df_to_mcap
from helpers.wrapper import ScaledModelWrapper, ModelSaver
from helpers.torch_model import TorchTransformerModel
from helpers.trainer import train


def main():
    # Configuration
    data_freq = 80  # Desired frequency in Hz
    stride = 4  # Stride between future steps (4 for 20Hz prediction from 80Hz data)
    inference_freq = data_freq // stride  # Inference frequency in Hz
    prediction = False  # Whether we are doing prediction or estimation
    history_size = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_cols = ["desired_position_rad_data", "measured_position_rad_data", "measured_velocity_rad_per_sec_data"]
    # input_cols = ["delta_position_rad_data", "measured_velocity_rad_per_sec_data"]
    output_cols = ["load_newton_data"]
    mcap_file_paths = [
        "/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-14_55_45_0.mcap",
        "/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-14_57_32_0.mcap",
        "/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-14_59_00_0.mcap",
        "/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-14_59_58_0.mcap",
        "/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-15_01_44_0.mcap",
        "/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-15_03_59_0.mcap",
        "/workspace/data/training_data/2026_01_30/rosbag2_2026_01_30-15_05_30_0.mcap",  # test
        "/workspace/data/training_data/2026_02_02/rosbag2_2026_02_02-16_50_51_0.mcap",
        "/workspace/data/training_data/2026_02_02/rosbag2_2026_02_02-16_52_21_0.mcap",
        "/workspace/data/training_data/2026_02_02/rosbag2_2026_02_02-16_55_56_0.mcap",
        "/workspace/data/training_data/2026_02_02/rosbag2_2026_02_02-16_58_13_0.mcap",  # test
        "/workspace/data/training_data/2026_02_02/rosbag2_2026_02_02-16_59_40_0.mcap",
        "/workspace/data/training_data/2026_02_12/rosbag2_2026_02_12-15_53_55_0.mcap", # with RL
        "/workspace/data/training_data/2026_02_12/rosbag2_2026_02_12-15_55_22_0.mcap", # zero

        "/workspace/data/training_data/2026_02_13/rosbag2_2026_02_13-12_48_26_0.mcap", # RL
        "/workspace/data/training_data/2026_02_13/rosbag2_2026_02_13-12_58_23_0.mcap", # RL

        "/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-16_48_31_0.mcap",
        "/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-16_51_17_0.mcap", # finger slow step test
        "/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-16_53_16_0.mcap",
        "/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-16_53_59_0.mcap",
        "/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-16_55_49_0.mcap",
        "/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-17_03_48_0.mcap",
        "/workspace/data/training_data/2026_02_25/rosbag2_2026_02_25-17_05_21_0.mcap", # weak spring slow step test

        "/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-07_19_08_0.mcap",
        "/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-07_25_25_0.mcap",
        "/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-07_27_34_0.mcap",
        "/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-07_29_45_0.mcap",
        # ("/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-07_17_29_0.mcap", None), # test data weak spring
        # ("/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-07_31_02_0.mcap", None), # test data strong spring
        # ("/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-07_34_26_0.mcap", None), # test data finger
        "/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-08_40_42_0.mcap", # 3s holds finger
        "/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-08_49_08_0.mcap", # 3s holds strong spring
        "/workspace/data/training_data/2026_02_26/rosbag2_2026_02_26-08_57_10_0.mcap", # 3s holds weak spring

        "/workspace/data/training_data/2026_02_27/rosbag2_2026_02_27-15_38_59_0.mcap",
        "/workspace/data/training_data/2026_02_27/rosbag2_2026_02_27-15_42_17_0.mcap",
        "/workspace/data/training_data/2026_02_28/rosbag2_2026_02_28-09_02_59_0.mcap",

        "/workspace/data/training_data/2026_02_28/mlp_30_model_steps_rosbag2_2026_02_28-09_30_03_0.mcap",
        "/workspace/data/training_data/2026_02_28/mlp_30_pure_rosbag2_2026_02_28-09_41_03_0.mcap",

        "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-13_21_08_0.mcap", 
        "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-13_44_47_0.mcap",
        "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-13_46_01_0.mcap",
        "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-14_05_04_0.mcap",

        # "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-15_35_25_0.mcap", # test for paper
        # "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-15_37_21_0.mcap", # test for paper
        # "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-15_38_00_0.mcap", # test for paper

        "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-16_50_04_0.mcap",
        "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-17_04_45_0.mcap",
        
        
    ]

    all_inputs = torch.empty((0, history_size, len(input_cols)), device=device)
    all_outputs = torch.empty((0, 1, len(output_cols)), device=device)
    for mcap_file_path in mcap_file_paths:
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=data_freq)
        data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()  # Remove duplicate timestamps by keeping the first occurrence
        process_dataframe(data_df_extrapolated)
        data_df_to_mcap(data_df_extrapolated, mcap_file_path.replace(".mcap", "_processed"))
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
        input_indices = [col_names.index(col) for col in input_cols]
        output_indices = [col_names.index(col) for col in output_cols]
        inputs = process_inputs_time_series(data_tensor[:, input_indices], stride=stride, history_size=history_size, prediction=prediction)
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
