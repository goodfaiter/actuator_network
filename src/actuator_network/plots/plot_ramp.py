import matplotlib.pyplot as plt

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe

# LINEWIDTH = 3 # 1 standard, 3 for overleaf
LABELS_FONT = 13  # 14 stadard, 30 for overleaf
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = LABELS_FONT

df = read_mcap_to_dataframe(
    "/workspace/src/actuator_network/plots/data/ramp_rosbag2_2026_02_27-14_16_41_0_processed_0.mcap",
    topics=["/desired_position_rad_data", "/measured_position_rad_data", "/load_newton_data"],
)
df = df.groupby(df.index).first()
df = df[500:1800]

time = [i * 0.0125 for i in range(len(df))]  # 0.0125 is the extrapolated frequency (80Hz)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), dpi=500)

ax1.plot(time, df["measured_position_rad_data_data"], label="Measured")
ax1.plot(time, df["desired_position_rad_data_data"], label="Desired")
# ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Position [rad]")
ax1.grid(True, alpha=0.3)
# remove x axis numbers but keep grid
ax1.set_xticklabels([])
ax1.legend()

ideal_pid = (df["desired_position_rad_data_data"] - df["measured_position_rad_data_data"]) * 4.3
ax2.plot(time, df["load_newton_data_data"], label="Measured")
ax2.plot(time, ideal_pid, label="Ideal", color="red")
# ax2.plot(time, trans_df["desired_ee_angle_rad_data_data"], label="Desired EE Angle")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Load [N]")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/workspace/src/actuator_network/plots/figures/ramp_tracking.png", dpi=500, bbox_inches="tight")
