from helpers.mcap_to_pandas import read_mcap_to_dataframe


def main():
    # Configuration
    mcap_file_paths = [
        "/workspace/data/training_data/2026_01_14/test.mcap", # test data, delete later
    ]

    data = read_mcap_to_dataframe(mcap_file_paths[0])

    print(data)


if __name__ == "__main__":
    main()
