import matplotlib.pyplot as plt
import numpy as np

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe

LINEWIDTH = 2  # 1 standard, 3 for overleaf
LABELS_FONT = 13  # 14 stadard, 30 for overleaf
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = LABELS_FONT

topics = [
    "/desired_position_rad_data",
    "/measured_position_rad_data",
    "/load_newton_data",
    "/load_newton_data_predicted",
]

df_unblocked = read_mcap_to_dataframe(
    "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-15_35_25_0_predicted/rosbag2_2026_03_03-15_35_25_0_predicted_0.mcap",
    topics=topics,
)
df_unblocked = df_unblocked.groupby(df_unblocked.index).first()
df_unblocked = df_unblocked[2 * 80 : 17 * 80]

df_half_blocked = read_mcap_to_dataframe(
    "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-15_37_21_0_predicted/rosbag2_2026_03_03-15_37_21_0_predicted_0.mcap",
    topics=topics,
)
df_half_blocked = df_half_blocked.groupby(df_half_blocked.index).first()
df_half_blocked = df_half_blocked[6 * 80 : 21 * 80]

df_blocked = read_mcap_to_dataframe(
    "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-15_38_00_0_predicted/rosbag2_2026_03_03-15_38_00_0_predicted_0.mcap",
    topics=topics,
)
df_blocked = df_blocked.groupby(df_blocked.index).first()
df_blocked = df_blocked[2 * 80 : 17 * 80]

# df = df + read_mcap_to_dataframe("/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-13_44_47_0_predicted/rosbag2_2026_03_03-13_44_47_0_predicted_0.mcap", topics=topics)  # noqa: E501
# df = df.groupby(df.index).first()
# df = df[150:2560]

# df = df_unblocked
# df = df_half_blocked
df = df_blocked

dfs = [df_unblocked, df_half_blocked, df_blocked]

for i, df in enumerate(dfs):
    time = [i * 0.0125 for i in range(len(df))]  # 0.0125 is the extrapolated frequency (80Hz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), dpi=500)

    ax1.plot(time, df["measured_position_rad_data_data"], label="Measured", linewidth=LINEWIDTH)
    ax1.plot(time, df["desired_position_rad_data_data"], label="Desired", linewidth=LINEWIDTH)
    ax1.set_ylabel("Position [rad]")
    ax1.grid(True, alpha=0.3)

    ideal_pid = (df["desired_position_rad_data_data"] - df["measured_position_rad_data_data"]) * 4.3
    ax2.plot(time, df["load_newton_data_data"], label="Measured", linewidth=LINEWIDTH)
    ax2.plot(time, ideal_pid, label="Ideal", color="red", linewidth=LINEWIDTH)
    ax2.plot(time, df["load_newton_data_predicted_data"], label="Predicted", color="green")
    ax2.set_ylabel("Load [N]")
    ax2.grid(True, alpha=0.3)

    rmse_measured = np.sqrt(np.mean((df["load_newton_data_data"] - df["load_newton_data_predicted_data"]) ** 2))
    print(f"RMSE Measured vs Predicted: {rmse_measured:.2f} N")

    if i == 0:
        ax1.legend()
        ax2.legend()

    if i < 2:
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])

    if i == 2:
        ax1.set_xlabel("Time [s]")
        ax2.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig(
        f"/workspace/src/actuator_network/plots/figures/contact_ramp_tracking_{i}.png", dpi=500, bbox_inches="tight"
    )
