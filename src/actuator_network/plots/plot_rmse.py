from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe

LINEWIDTH = 3  # 1 standard, 3 for overleaf
LABELS_FONT = 16  # 14 stadard, 30 for overleaf
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = LABELS_FONT


def plot_minimap(des):
    plt.figure(figsize=(10, 1.5), dpi=500)

    time_axis = np.arange(len(des)) * 1 / 80

    plt.plot(
        time_axis, des, marker="", linestyle="-", label="Desired Position [rad]", color="red", alpha=0.8, markersize=4
    )

    # Add labels and title
    # plt.xlabel('Time [s]')
    # plt.ylabel('Desired Position [rad]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig("/workspace/src/actuator_network/plots/figures/rmse_minimap.png", dpi=500, bbox_inches="tight")


def plot(dfs: list, labels: list, file_prefix: str = "", title: Optional[str] = None, legend: bool = True):
    plt.figure(figsize=(7, 5), dpi=500)

    time_axis = np.arange(len(dfs[0])) * 1 / 80

    for df, label in zip(dfs, labels):
        plt.plot(
            time_axis,
            df["load_newton_data_predicted_data"],
            label=f"{label}",
            marker="",
            linestyle="-",
            alpha=0.8,
            markersize=4,
        )

    plt.plot(time_axis, dfs[0]["load_newton_data_data"], label="Measured", marker="", linestyle="-", markersize=4)

    # Add labels and title
    plt.xlabel("Time [s]")
    plt.ylabel("Tendon Force [N]")
    if title:
        plt.title(title)
    if legend:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        f"/workspace/src/actuator_network/plots/figures/{file_prefix}_actual_vs_predicted.png",
        dpi=500,
        bbox_inches="tight",
    )


def plot_bars(
    nums: Dict[str, List[float]],
    stds: Dict[str, List[float]],
    labels: List[str],
    file_prefix: str = "",
    title: Optional[str] = None,
):
    """
    Create grouped bar plots.

    nums: dict where each key is a group name, value is list of bar heights
    stds: dict where each key is a group name, value is list of standard deviations
    labels: labels for each bar position
    file_prefix: optional filename prefix to save plot
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=500)

    x = np.arange(len(nums))  # positions for groups (dict keys)
    width = 0.8 / len(labels)  # width of each bar

    # Plot each label as a separate set of bars across groups
    for i, label in enumerate(labels):
        values = [nums[key][i] for key in nums.keys()]
        errors = [stds[key][i] for key in nums.keys()]
        offset = (i - len(labels) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, yerr=errors, capsize=3)

    ax.set_xticks(x)
    ax.set_ylabel("RMSE [N]")
    ax.set_xticklabels(nums.keys())
    if title:
        ax.set_title(title)
    ax.legend()

    plt.savefig(f"/workspace/src/actuator_network/plots/figures/{file_prefix}_bars.png", dpi=500, bbox_inches="tight")


range = [0, -1]
bars = {"weak": [], "strong": [], "finger": []}
bars_mean_abs_error = {"weak": [], "strong": [], "finger": []}
bars_std = {"weak": [], "strong": [], "finger": []}

### weak
# weak_range = [0, 80 * 9]
# file_prefix = "weak_spring_1"
# weak_range = [80 * 9, 80 * 18]
# file_prefix = "weak_spring_2"
# weak_range = [80 * 17, -1]
# file_prefix = "weak_spring_3"
weak_range = range
file_prefix = "weak_spring_full"
mcap_file_paths = [
    "/workspace/src/actuator_network/plots/data/transformer_30/rosbag2_2026_02_26-09_23_45_0_predicted/rosbag2_2026_02_26-09_23_45_0_predicted_0.mcap",
    "/workspace/src/actuator_network/plots/data/mlp_30/rosbag2_2026_02_26-09_23_45_0_predicted/rosbag2_2026_02_26-09_23_45_0_predicted_0.mcap",
    "/workspace/src/actuator_network/plots/data/rnn/rosbag2_2026_02_26-09_23_45_0_predicted/rosbag2_2026_02_26-09_23_45_0_predicted_0.mcap",
]

data_dfs = []
first = True
for mcap_file_path in mcap_file_paths:
    data_df = read_mcap_to_dataframe(
        mcap_file_path, topics=["/desired_position_rad_data", "/load_newton_data", "/load_newton_data_predicted"]
    )
    data_df = data_df.groupby(data_df.index).first()
    data_df = data_df.iloc[weak_range[0] : weak_range[1]]
    # print(data_df.head())
    test = data_df["desired_position_rad_data_data"][0:-1]
    if first:
        plot_minimap(test)
        first = False
    rmse = ((data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]) ** 2).mean() ** 0.5
    mean_abs_error = (data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]).abs().mean()
    std = abs(data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]).std()
    print(f"RMSE: {rmse}, STD: {std}, MEAN ABS ERROR: {mean_abs_error}")
    bars["weak"].append(rmse)
    bars_mean_abs_error["weak"].append(mean_abs_error)
    bars_std["weak"].append(std)
    data_dfs.append(data_df)
plot(data_dfs, labels=["Transformer", "MLP", "RNN"], file_prefix=file_prefix, title="Weak Spring", legend=False)

### strong
# strong_range = [0, 80 * 9]
# file_prefix = "strong_spring_1"
# strong_range = [80 * 11, 80 * 20]
# file_prefix = "strong_spring_2"
# strong_range = [80 * 20, -1]
# file_prefix = "strong_spring_3"
strong_range = range
file_prefix = "strong_spring_full"
mcap_file_paths = [
    "/workspace/src/actuator_network/plots/data/transformer_30/rosbag2_2026_02_26-09_17_44_0_predicted/rosbag2_2026_02_26-09_17_44_0_predicted_0.mcap",
    "/workspace/src/actuator_network/plots/data/mlp_30/rosbag2_2026_02_26-09_17_44_0_predicted/rosbag2_2026_02_26-09_17_44_0_predicted_0.mcap",
    "/workspace/src/actuator_network/plots/data/rnn/rosbag2_2026_02_26-09_17_44_0_predicted/rosbag2_2026_02_26-09_17_44_0_predicted_0.mcap",
]

data_dfs = []
for mcap_file_path in mcap_file_paths:
    data_df = read_mcap_to_dataframe(
        mcap_file_path, topics=["/desired_position_rad_data", "/load_newton_data", "/load_newton_data_predicted"]
    )
    data_df = data_df.groupby(data_df.index).first()
    data_df = data_df.iloc[strong_range[0] : strong_range[1]]
    rmse = ((data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]) ** 2).mean() ** 0.5
    mean_abs_error = (data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]).abs().mean()
    std = abs(data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]).std()
    print(f"RMSE: {rmse}, STD: {std}, MEAN ABS ERROR: {mean_abs_error}")
    bars["strong"].append(rmse)
    bars_mean_abs_error["strong"].append(mean_abs_error)
    bars_std["strong"].append(std)
    data_dfs.append(data_df)
plot(data_dfs, labels=["Transformer", "MLP", "RNN"], file_prefix=file_prefix, title="Strong Spring", legend=False)

### finger
# finger_range = [0, 80 * 11]
# file_prefix = "finger_1"
# finger_range = [int(80 * 11.5), int(80 * 20.5)]
# file_prefix = "finger_2"
# finger_range = [80 * 23, -1]
# file_prefix = "finger_3"
finger_range = range
file_prefix = "finger_full"
mcap_file_paths = [
    "/workspace/src/actuator_network/plots/data/transformer_30/rosbag2_2026_02_26-09_29_17_0_predicted/rosbag2_2026_02_26-09_29_17_0_predicted_0.mcap",
    "/workspace/src/actuator_network/plots/data/mlp_30/rosbag2_2026_02_26-09_29_17_0_predicted/rosbag2_2026_02_26-09_29_17_0_predicted_0.mcap",
    "/workspace/src/actuator_network/plots/data/rnn/rosbag2_2026_02_26-09_29_17_0_predicted/rosbag2_2026_02_26-09_29_17_0_predicted_0.mcap",
]

data_dfs = []
for mcap_file_path in mcap_file_paths:
    data_df = read_mcap_to_dataframe(
        mcap_file_path, topics=["/desired_position_rad_data", "/load_newton_data", "/load_newton_data_predicted"]
    )
    data_df = data_df.groupby(data_df.index).first()
    data_df = data_df.iloc[finger_range[0] : finger_range[1]]
    rmse = ((data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]) ** 2).mean() ** 0.5
    mean_abs_error = abs(data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]).mean()
    std = abs(data_df["load_newton_data_data"] - data_df["load_newton_data_predicted_data"]).std()
    print(f"RMSE: {rmse}, STD: {std}, MEAN ABS ERROR: {mean_abs_error}")
    bars["finger"].append(rmse)
    bars_mean_abs_error["finger"].append(mean_abs_error)
    bars_std["finger"].append(std)

    data_dfs.append(data_df)
plot(data_dfs, labels=["Transformer", "MLP", "RNN"], file_prefix=file_prefix, title="Finger", legend=True)

# plot bars with error std as error bars
title = "Total"
plot_bars(bars, bars_std, labels=["Transformer", "MLP", "RNN"], file_prefix="rmse_comparison", title=title)
plot_bars(bars_mean_abs_error, bars_std, labels=["Transformer", "MLP", "RNN"], file_prefix="mean_abs_error_comparison")

# RMSE: 0.538697157593638
# RMSE: 0.7234664814835885
# RMSE: 0.8860167042285075
# RMSE: 0.6873743864738324
# RMSE: 0.9516271263113869
# RMSE: 0.9048958534134514
# RMSE: 0.8047928270698547
# RMSE: 0.9239295626671089
# RMSE: 1.9381296076867691
