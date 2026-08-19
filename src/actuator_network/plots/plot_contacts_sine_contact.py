from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

LINEWIDTH = 2  # 1 standard, 3 for overleaf
LABELS_FONT = 13  # 14 stadard, 30 for overleaf
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = LABELS_FONT

topics = ["/desired_position_rad_data", "/measured_position_rad_data", "/load_newton_data", "/load_newton_data_predicted"]

df = read_mcap_to_dataframe(
    "/workspace/data/training_data/2026_03_03/rosbag2_2026_03_03-17_05_48_0_predicted/rosbag2_2026_03_03-17_05_48_0_predicted_0.mcap",
    topics=topics,
)
df = df.groupby(df.index).first()
df = df[5*80: 13*80]
# df = df[0 : -1]

time = [i * 0.0125 for i in range(len(df))]  # 0.0125 is the extrapolated frequency (80Hz)

fig, (ax2) = plt.subplots(1, 1, figsize=(10, 4), dpi=500)

# ax1.plot(time, df["measured_position_rad_data_data"], label="Measured", linewidth=LINEWIDTH)
# ax1.plot(time, df["desired_position_rad_data_data"], label="Desired", linewidth=LINEWIDTH)
# ax1.set_ylabel("Position [rad]")
# ax1.legend()
# ax1.set_xlabel("Time [s]")
# ax1.grid(True, alpha=0.3)

ideal_pid = (df["desired_position_rad_data_data"] - df["measured_position_rad_data_data"]) * 4.3
ax2.plot(time, df["load_newton_data_data"], label="Measured", linewidth=LINEWIDTH)
ax2.plot(time, ideal_pid, label="Ideal", color="red", linewidth=LINEWIDTH)
ax2.plot(time, df["load_newton_data_predicted_data"], label="Predicted", color="green")
ax2.set_ylabel("Load [N]")

# plots the plots error on the right y-axis
error = df["measured_position_rad_data_data"]
ax2_twin = ax2.twinx()
ax2_twin.plot(time, error, label="Motor Position", linestyle="--", color="grey", alpha=0.5, linewidth=LINEWIDTH)
ax2_twin.set_ylabel("Motor Position [rad]")
ax2_twin.tick_params(axis="y")
ax2_twin.legend(loc="upper right")

ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Time [s]")

rmse_measured = np.sqrt(np.mean((df["load_newton_data_data"] - df["load_newton_data_predicted_data"]) ** 2))
print(f"RMSE Measured vs Predicted: {rmse_measured:.2f} N")

plt.tight_layout()
plt.savefig(f"/workspace/src/actuator_network/plots/figures/contact_ramp_tracking_sine.png", dpi=500, bbox_inches="tight")
