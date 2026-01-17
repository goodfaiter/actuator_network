from helpers.mcap_to_pandas import read_mcap_to_dataframe
from helpers.pandas_processing import extrapolate_dataframe

def main():
    # Configuration
    mcap_file_paths = [
        "/workspace/data/training_data/2026_01_17/rosbag2_2026_01_17_14_37_22_0_recovered.mcap", # test data, delete later
    ]

    data = read_mcap_to_dataframe(mcap_file_paths[0])
    data = extrapolate_dataframe(data, freq=80)

    print(data)

    #plot data for testing
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 8))
    # for column in data.columns:
    #     plt.plot(data.index, data[column], label=column)
    # plt.xlabel("Time")
    # plt.ylabel("Values")
    # plt.title("Extrapolated Data from MCAP")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
