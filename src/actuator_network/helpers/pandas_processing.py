import pandas as pd


def extrapolate_dataframe(df: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Resample dataframe to fixed frequency with proper interpolation"""

    # Create target index at the exact desired frequency.
    target_period = pd.Timedelta(seconds=1.0 / freq)

    # Resample directly to the target frequency and interpolate missing values.
    df_extrapolated = df.resample(target_period).mean().interpolate(method="linear", limit_direction="both")

    df_extrapolated.index = df_extrapolated.index - df_extrapolated.index[0]

    return df_extrapolated


def filter_signal(signal: pd.Series, alpha: float = 0.1) -> pd.Series:
    """Apply a simple low-pass filter to the signal using exponential moving average."""
    filtered_signal = signal.ewm(alpha=alpha).mean()
    return filtered_signal


def derivate_signal(signal: pd.Series, dt: float) -> pd.Series:
    """Calculate the derivative of the signal."""
    derivative = signal.diff().fillna(0) / dt
    return derivative


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the tendon force from weight and acceleration"""
    mass = 0.03  # kg
    g = 9.81  # m/s^2
    radius = 0.011  # m

    df["delta_position_rad_data"] = df["desired_position_rad_data"] - df["measured_position_rad_data"]

    df["calculated_velocity_meter_per_sec_data"] = df["measured_velocity_rad_per_sec_data"] * radius

    # Acceleration of motor
    df["calculated_acceleration_meter_per_sec2_data"] = (
        derivate_signal(df["measured_velocity_rad_per_sec_data"], dt=1 / 80) * radius
    )
    df["calculated_dynamic_force_newton_data"] = df["calculated_acceleration_meter_per_sec2_data"] * mass

    # weight_offset = df["weight_kg_data"][:num_samples_for_offset].mean()
    # acceleration_offset_x = df["imu_data_raw_linear_acceleration_x"][:num_samples_for_offset].mean()
    # acceleration_offset_y = df["imu_data_raw_linear_acceleration_y"][:num_samples_for_offset].mean()
    # acceleration_offset_z = df["imu_data_raw_linear_acceleration_z"][:num_samples_for_offset].mean()

    # df["weight_kg_data"] = df["weight_kg_data"] - weight_offset
    df["weight_kg_data"] = df["weight_kg_data"]
    df["load_newton_data"] = df["weight_kg_data"] * g
    # df["imu_data_raw_linear_acceleration_x"] = df["imu_data_raw_linear_acceleration_x"] - acceleration_offset_x
    # df["imu_data_raw_linear_acceleration_y"] = df["imu_data_raw_linear_acceleration_y"] - acceleration_offset_y
    # df["imu_data_raw_linear_acceleration_z"] = df["imu_data_raw_linear_acceleration_z"] - acceleration_offset_z

    # filter acceleration
    # df["imu_data_raw_linear_acceleration_x_filtered"] = filter_signal(df["imu_data_raw_linear_acceleration_x"])
    # df["imu_data_raw_linear_acceleration_y_filtered"] = filter_signal(df["imu_data_raw_linear_acceleration_y"])
    # df["imu_data_raw_linear_acceleration_z_filtered"] = filter_signal(df["imu_data_raw_linear_acceleration_z"])

    # df["measured_acceleration"] = (
    #     df["imu_data_raw_linear_acceleration_x_filtered"] ** 2
    #     + df["imu_data_raw_linear_acceleration_y_filtered"] ** 2
    #     + df["imu_data_raw_linear_acceleration_z_filtered"] ** 2
    # )
    # df["measured_acceleration"] = df["measured_acceleration"].pow(0.5)

    df["tendon_force_newton_data"] = df["load_newton_data"] + df["calculated_dynamic_force_newton_data"]

    df["tendon_bota_force_newton_data"] = df["bota_wrench_N_and_Nm_torque_z"] / radius

    # reindex with new col
    df = df.reindex(columns=[*df.columns.tolist()])

    return df
