import pandas as pd


def extrapolate_dataframe(df: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Resample dataframe to fixed frequency with proper interpolation"""

    # Create target index at the exact desired frequency.
    target_period = pd.Timedelta(seconds=1.0 / freq)

    # Resample directly to the target frequency and interpolate missing values.
    df_extrapolated = df.resample(target_period).mean().interpolate(method="linear", limit_direction="both")

    df_extrapolated.index = df_extrapolated.index - df_extrapolated.index[0]

    return df_extrapolated


def derivate_signal(signal: pd.Series, dt: float) -> pd.Series:
    """Calculate the derivative of the signal."""
    derivative = signal.diff().fillna(0) / dt
    return derivative


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the tendon force from weight and acceleration.

    ``df`` must be resampled to a fixed cadence (e.g. via :func:`extrapolate_dataframe`)
    so that the timestep ``dt`` used for the derivative can be derived from the index spacing.
    """
    mass = 0.03  # kg
    radius = 0.011  # m
    dt = (df.index[1] - df.index[0]).total_seconds() if len(df.index) > 1 else 1.0

    df["delta_position_rad_data"] = df["desired_position_rad_data"] - df["measured_position_rad_data"]

    df["calculated_velocity_meter_per_sec_data"] = df["measured_velocity_rad_per_sec_data"] * radius

    # Acceleration of motor
    df["calculated_acceleration_meter_per_sec2_data"] = (
        derivate_signal(df["measured_velocity_rad_per_sec_data"], dt=dt) * radius
    )
    df["calculated_dynamic_force_newton_data"] = df["calculated_acceleration_meter_per_sec2_data"] * mass

    df["tendon_bota_force_newton_data"] = df["bota_wrench_N_and_Nm_torque_z"] / radius

    return df
