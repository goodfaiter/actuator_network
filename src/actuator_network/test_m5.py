"""Run M5 friction envelope inference on test MCAPs.

Loads the fitted M5 envelope model (``m5_model.pt``, saved by ``train-m5``),
predicts the friction envelope for every preprocessed sample and writes
prediction MCAPs with the ``friction_envelope_Nm_predicted`` column.

Usage:

.. code-block:: bash

    uv run test-m5

"""

import json

import torch

from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel_cached
from actuator_network.helpers.hyperparameters import M5FrictionConfig
from actuator_network.helpers.pandas_processing import process_dataframe
from actuator_network.helpers.pandas_to_mcap import data_df_to_mcap
from actuator_network.helpers.torch_model import M5EnvelopeFrictionModel
from actuator_network.train_m5 import (
    TAU_EXTERNAL_COL,
    TAU_MOTOR_COL,
    VELOCITY_COL,
    build_friction_samples,
    envelope_losses,
    friction_sample_kwargs,
)

DEFAULT_MODEL_PATH = "/workspace/data/output_data/m5_model.pt"

FRICTION_ENVELOPE_COL = "friction_envelope_Nm_predicted"
OUTPUT_SUFFIX = "_m5_predicted"


def load_m5_model(model_path: str, device: torch.device) -> M5EnvelopeFrictionModel:
    """Load the fitted M5 envelope model from its saved state dict.

    The params JSON saved next to the model provides the fixed-parameter set
    so that the state dict with fixed buffers loads strictly.

    Args:
        model_path: Path to the saved model state dict (``m5_model.pt``).
        device: Torch device.

    Returns:
        The loaded model in eval mode.
    """
    with open(model_path.replace("model.pt", "params.json")) as f:
        saved = json.load(f)
    fixed = {name: saved[name] for name in saved.get("fixed", [])}
    init = {name: value for name, value in saved.items() if name in M5EnvelopeFrictionModel.PARAM_NAMES and name not in fixed}
    model = M5EnvelopeFrictionModel(init_params=init, device=device, fixed_params=fixed)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _print_envelope_metrics(model: M5EnvelopeFrictionModel, df, config: M5FrictionConfig, device: torch.device) -> None:
    """Print the envelope fit metrics for one recording (train_m5 loss definitions)."""
    samples = build_friction_samples([df], **friction_sample_kwargs(config, device))
    with torch.no_grad():
        losses = {k: v.item() for k, v in envelope_losses(model, samples, config).items()}
    print(
        f"  moving RMSE: {losses['moving_mse'] ** 0.5:.6f} Nm, breakaway RMSE: {losses['breakaway_mse'] ** 0.5:.6f} Nm, "
        f"static bound: {losses['static_bound']:.3e} Nm^2, static tightness: {losses['static_tightness']:.3e} Nm^2"
    )


def run_m5_inference(
    model_path: str,
    mcap_file_paths: list[str],
    config: M5FrictionConfig | None = None,
) -> list[str]:
    """Run M5 envelope inference on the given MCAPs and write prediction MCAPs.

    The preprocessing and sample-type thresholds match ``train_m5``. The model
    is stateless and element-wise, so the envelope is predicted for every
    preprocessed sample in one forward pass per recording.

    Args:
        model_path: Path to the saved model state dict (``m5_model.pt``).
        mcap_file_paths: List of input MCAP files.
        config: Hyperparameter configuration; defaults to ``M5FrictionConfig()``.

    Returns:
        List of output file paths.
    """
    config = config or M5FrictionConfig()
    device = torch.device("cpu")

    print("Loading M5 envelope model...")
    model = load_m5_model(model_path, device)

    print("Loading and processing MCAP files...")
    dataframes = load_mcap_dataframes_parallel_cached(mcap_file_paths, freq=config.data_freq)
    # The parquet cache is not keyed on the processing code; recompute derived torque columns.
    for df in dataframes:
        process_dataframe(df)

    output_paths = []
    for mcap_file_path, df in zip(mcap_file_paths, dataframes):
        with torch.no_grad():
            envelope = model(
                torch.as_tensor(df[VELOCITY_COL].to_numpy(dtype="float32"), device=device),
                torch.as_tensor(df[TAU_MOTOR_COL].to_numpy(dtype="float32"), device=device),
                torch.as_tensor(df[TAU_EXTERNAL_COL].to_numpy(dtype="float32"), device=device),
            ).numpy()
        df[FRICTION_ENVELOPE_COL] = envelope

        _print_envelope_metrics(model, df, config, device)

        output_path = mcap_file_path.replace(".mcap", f"{OUTPUT_SUFFIX}.mcap")
        data_df_to_mcap(df, output_path)
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

    run_m5_inference(DEFAULT_MODEL_PATH, mcap_file_paths)


if __name__ == "__main__":
    main()
