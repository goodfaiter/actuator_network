"""Run inference with the spring + force transformer model."""

import torch

from actuator_network.helpers.mcap_to_pandas import read_mcap_to_dataframe
from actuator_network.helpers.pandas_processing import extrapolate_dataframe, process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.pandas_to_torch import pandas_to_torch

DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_spring_transformer_latest.pt"


def run_spring_transformer_inference(
    model_path: str,
    mcap_file_paths: list[str],
    output_suffix: str = "_spring_transformer_predicted",
    require_pruned: bool = False,
) -> list[str]:
    """Run spring/force transformer inference on the given MCAPs.

    One model call is one force tick, so inference is performed at the model's
    inference rate (every ``stride``-th preprocessed sample) by feeding the
    latest normalized sample; the model maintains its internal buffers. A fresh
    reset is performed per recording.

    Both checkpoint types are accepted: the full adaptive checkpoint
    (``best_spring_transformer_latest.pt``) and the pruned frozen-latent
    deployment checkpoints (``spring_transformer_frozen_<label>.pt``, see
    ``actuator_network.export_frozen_latent``), whose embedded latent pins the
    force transformer to one specific spring coefficient.

    Args:
        model_path: Path to the saved TorchScript model.
        mcap_file_paths: List of input MCAP files.
        output_suffix: Suffix of the written ``*_predicted.mcap`` outputs.
        require_pruned: When True, the checkpoint must be a pruned frozen-latent
            deployment model (used by the dedicated ``test-spring-transformer-frozen``
            entry point).

    Returns:
        List of output file paths.
    """
    device = torch.device("cpu")

    print("Loading spring transformer model...")
    model = torch.jit.load(model_path, map_location=device)

    data_freq = model.metadata["frequency"]
    if data_freq <= 1:
        raise ValueError(f"Checkpoint stores an invalid frequency: {data_freq}")

    if require_pruned:
        if hasattr(model.model, "model_transformer"):
            raise ValueError(
                "This entry requires the pruned frozen-latent deployment checkpoints "
                "spring_transformer_frozen_<label>.pt (exported by `uv run export-frozen-latent`); "
                "the full adaptive checkpoint is not a frozen deployment model."
            )
        if not hasattr(model.model, "frozen_latent"):
            raise ValueError("Checkpoint is not a spring transformer frozen deployment model.")
        if float(model.model.frozen_latent.abs().max().item()) == 0.0:
            raise ValueError(
                "The pruned checkpoint stores an all-zero frozen latent; re-export with `uv run export-frozen-latent`."
            )

    stride = model.metadata["stride"]
    input_cols = model.input_columns
    output_cols = model.output_columns

    output_paths = []
    for mcap_file_path in mcap_file_paths:
        model.reset()
        data_df = read_mcap_to_dataframe(mcap_file_path)
        data_df_extrapolated = extrapolate_dataframe(data_df, freq=data_freq)
        data_df_extrapolated = data_df_extrapolated.groupby(data_df_extrapolated.index).first()
        process_dataframe(data_df_extrapolated)
        col_names, data_tensor = pandas_to_torch(data_df_extrapolated, device=device)
        input_indices = [col_names.index(col) for col in input_cols]
        features = data_tensor[:, input_indices]

        for col in output_cols:
            data_df_extrapolated[col + "_predicted"] = 0.0

        num_samples = features.shape[0]
        predictions = torch.zeros((num_samples, len(output_cols)))

        # One model call is one force tick: call at the inference rate by
        # stepping over the preprocessed samples with the model's stride, and
        # feed only the latest sample of shape [1, 1, F].
        for t in range(0, num_samples, stride):
            sample = features[t].view(1, 1, -1)
            with torch.no_grad():
                pred = model(sample)
            predictions[t, :] = pred[0, 0, :]

        for i, col in enumerate(output_cols):
            data_df_extrapolated[col + "_predicted"] = predictions[:, i].numpy()

        # The predicted recording holds the samples the model was called at:
        # downsampling keeps only the inferred rows, so the output .mcap is at
        # the model's inference frequency (data_freq / stride).
        predicted_df = data_df_extrapolated.iloc[::stride]

        output_path = mcap_file_path.replace(".mcap", f"{output_suffix}.mcap")
        data_df_to_mcap(predicted_df, output_path)
        print(f"  wrote {output_path}")
        output_paths.append(output_path)

    return output_paths


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_16_16_0.mcap",  # blocked
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_46_40_0.mcap",  # strong spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_27_33_0.mcap",  # weak spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_19_09_0.mcap",  # finger
    ]

    run_spring_transformer_inference(DEFAULT_MODEL_PATH, mcap_file_paths)


if __name__ == "__main__":
    main()
