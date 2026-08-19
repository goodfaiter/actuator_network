import matplotlib.pyplot as plt

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe

LINEWIDTH = 2  # 1 standard, 3 for overleaf
LABELS_FONT = 16  # 14 stadard, 30 for overleaf
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = LABELS_FONT

start = 0
offset = 280
length = 1500
pure_df = read_mcap_to_dataframe(
    "/workspace/src/actuator_network/plots/data/mlp_30_pure_rosbag2_2026_02_28-09_41_03_0_processed_0.mcap",
    topics=["/measured_position_rad_data"],
)
pure_df = pure_df.groupby(pure_df.index).first()
pure_df = pure_df.iloc[start : start + length]

trans_df = read_mcap_to_dataframe(
    "/workspace/src/actuator_network/plots/data/mlp_30_model_steps_rosbag2_2026_02_28-09_30_03_0_processed_0.mcap",
    topics=["/measured_position_rad_data", "/desired_ee_angle_rad_data"],
)
trans_df = trans_df.groupby(trans_df.index).first()
trans_df = trans_df.iloc[start + offset : start + offset + length]

time = [i * 0.0125 for i in range(length)]  # 0.0125 is the extrapolated frequency (80Hz)

fig, (ax2) = plt.subplots(1, 1, figsize=(8, 2.5))

# ax1.plot(time, trans_df["desired_ee_angle_rad_data_data"], label="Desired Finger Angle", color="green", linewidth=LINEWIDTH)  # noqa: E501
# # ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Desired Angle [rad]")
# ax1.grid(True, alpha=0.3)
# # remove x axis numbers but keep grid
# ax1.set_xticklabels([])
# ax1.legend()

ax2.plot(time, pure_df["measured_position_rad_data_data"], label="Ideal", linewidth=LINEWIDTH)
ax2.plot(time, trans_df["measured_position_rad_data_data"], label="Transformer", linewidth=LINEWIDTH)
# ax2.plot(time, trans_df["desired_ee_angle_rad_data_data"], label="Desired EE Angle")
# ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Motor Position [rad]")
ax2.legend()
ax2.set_xticklabels([])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/workspace/src/actuator_network/plots/figures/ee_tracking_comparison.png", dpi=500)
