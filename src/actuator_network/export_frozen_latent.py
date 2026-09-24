"""Export frozen latent vectors and pruned deployment models from datasets of
single spring coefficients.

For a dataset of one specific spring coefficient this script computes the mean
latent representation (model-output space) over the dataset's frozen spring
windows — built exactly as during training (zero-padded aligned windows at the
checkpoint's spring history/stride, buffer frozen while the velocity stays
within the checkpoint's threshold bounds) — and saves it as a small ``.pt``
payload (an exported record of the latent: value, input columns, window
count, and the source checkpoint). That latent is additionally embedded into a
pruned deployment-only model (``FrozenLatentForceModel``, see
``helpers/torch_model.py``) built from the checkpoint's compiled force
transformer and spring coefficient head; the scaled scripted model is saved as
``spring_transformer_frozen_<label>.pt``. That checkpoint contains strictly
the components the frozen-latent deployment uses (no spring transformer, no
spring buffer, no EMA/anchor/counter/gating state), so deployment loads less
memory, runs only the force path, and always receives a latent permanently
pinned to that specific spring coefficient.

Both artifacts share the full model's normalization statistics and metadata.
All model configuration is read from the saved TorchScript checkpoint itself
(input columns, normalization statistics, frequency, spring history/stride,
velocity thresholds), so the exported latent is guaranteed consistent with the
deployed model. The export requires the full spring transformer checkpoint
(not an already-pruned one).

Usage:

.. code-block:: bash

    uv run export-frozen-latent

"""

import torch

from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel_cached
from actuator_network.helpers.pandas_to_torch import (
    apply_normalization,
    build_frozen_spring_windows,
    build_strided_windows,
    pandas_to_torch,
)
from actuator_network.helpers.torch_model import FrozenLatentForceModel
from actuator_network.helpers.wrapper import ScaledModelWrapper

DEFAULT_MODEL_PATH = "/workspace/data/output_data/best_spring_transformer_latest.pt"
OUTPUT_DIR = "/workspace/data/output_data/"
LATENT_BATCH = 2048

# One dataset group per specific spring coefficient; every group is exported to
# its own ``frozen_latent_<label>.pt`` file.
INPUT_DATASETS: dict[str, list[str]] = {
    "blocked": [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_01_21_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_03_39_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_05_41_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_13_54_0.mcap",
    ],
    "strong_spring": [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_38_20_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_39_52_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_42_57_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_44_39_0.mcap",
    ],
    "weak_spring": [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_18_11_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_20_54_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_23_21_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_24_47_0.mcap",
    ],
    "finger": [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_11_54_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_13_57_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_15_28_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_17_53_0.mcap",
    ],
}


def compute_frozen_latent(model: torch.jit.ScriptModule, mcap_file_paths: list[str]) -> tuple[torch.Tensor, int]:
    """Compute the mean latent over the frozen spring windows of the given MCAPs.

    The windows are built exactly as during training: zero-padded aligned
    windows at the checkpoint's spring history size / stride (normalized
    features), with the buffer frozen while the velocity stays within the
    checkpoint's threshold bounds. The model transformer is run once over the
    windows in batches and the latents are averaged to a single vector.

    Args:
        model: Loaded TorchScript spring transformer wrapper
            (``ScaledModelWrapper`` around ``SpringTransformerModel``).
        mcap_file_paths: Input MCAP files of one specific spring coefficient.

    Returns:
        Tuple of (frozen latent of shape [1, 1, latent_dim], number of windows).
    """
    device = torch.device("cpu")

    data_freq = model.metadata["frequency"]
    if data_freq <= 1:
        raise ValueError(f"Checkpoint stores an invalid frequency: {data_freq}")

    input_cols = model.input_columns
    inner = model.model
    velocity_idx = inner.velocity_idx
    spring_history_size = inner.spring_history_size
    spring_stride = int(inner.spring_stride)
    threshold_lo = float(inner.velocity_threshold_lo)
    threshold_hi = float(inner.velocity_threshold_hi)

    dataframes = load_mcap_dataframes_parallel_cached(mcap_file_paths, freq=data_freq)

    all_spring_windows = []
    for df in dataframes:
        col_names, data_tensor = pandas_to_torch(df, device=device)
        input_indices = [col_names.index(col) for col in input_cols]
        features = apply_normalization(data_tensor[:, input_indices], model.input_mean, model.input_std)

        spring_windows_raw = build_strided_windows(features, spring_history_size, spring_stride)
        all_spring_windows.append(
            build_frozen_spring_windows(
                spring_windows_raw,
                velocity_idx=velocity_idx,
                threshold_lo=threshold_lo,
                threshold_hi=threshold_hi,
            )
        )

    spring_windows = torch.cat(all_spring_windows, dim=0)
    num_windows = spring_windows.size(0)
    if num_windows == 0:
        raise ValueError(f"No windows could be built from the given MCAPs: {mcap_file_paths}")

    # Run the spring transformer over the windows in batches and average the
    # resulting latents into the single frozen-latent vector.
    latents = []
    with torch.no_grad():
        for start in range(0, num_windows, LATENT_BATCH):
            batch = spring_windows[start : start + LATENT_BATCH].to(device)
            latents.append(inner.model_transformer(batch))  # [batch, 1, latent_dim]

    frozen_latent = torch.cat(latents, dim=0).mean(dim=0, keepdim=False).reshape(1, 1, -1)
    return frozen_latent, num_windows


def build_frozen_latent_model(
    model: torch.jit.ScriptModule,
    frozen_latent: torch.Tensor,
) -> FrozenLatentForceModel:
    """Build the pruned deployment-only model from a spring transformer checkpoint.

    Only the components the frozen-latent path uses are kept: the compiled
    force transformer and spring coefficient head of the checkpoint, a stateful
    force buffer, and the frozen latent (embedded via ``set_frozen_latent``).
    The spring transformer, spring buffer, EMA state, update counter and
    velocity gating are dropped.

    Args:
        model: Loaded TorchScript spring transformer wrapper
            (``ScaledModelWrapper`` around ``SpringTransformerModel``) — the
            full checkpoint, not an already-pruned one.
        frozen_latent: Normalized latent vector of shape [1, 1, latent_dim].

    Returns:
        The eager ``FrozenLatentForceModel`` with the latent embedded.
    """
    inner = model.model
    if not hasattr(inner, "model_transformer"):
        raise ValueError(
            "Checkpoint is already pruned (frozen-latent only); "
            "the export requires the full spring transformer checkpoint."
        )

    pruned_model = FrozenLatentForceModel(
        force_transformer=inner.force_transformer,  # already-compiled submodule
        spring_coeff_head=inner.spring_coeff_head,
        latent_dim=int(inner.latent_dim),
        input_size=int(inner.model_transformer.input_size),
        force_history_size=int(inner.force_history_size),
    )
    pruned_model.set_frozen_latent(frozen_latent)
    return pruned_model


def export_frozen_deployment(
    model_path: str,
    datasets: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    """Export a frozen latent payload and a pruned deployment model per dataset group.

    Per label the following artifacts are written to the output directory:

    - ``frozen_latent_<label>.pt``: ``{"latent", "input_columns",
      "num_windows", "checkpoint"}`` payload; an exported record of the
      computed latent (the same value as the embedded one).
    - ``spring_transformer_frozen_<label>.pt``: pruned deployment-only scripted
      model (``ScaledModelWrapper`` around ``FrozenLatentForceModel``) with the
      frozen latent embedded; loads less memory and runs only the force path.
      Usable directly with ``run_spring_transformer_inference``.

    Args:
        model_path: Path to the saved TorchScript spring transformer wrapper
            (full checkpoint).
        datasets: Mapping of label to the input MCAP files of one specific
            spring coefficient.

    Returns:
        Mapping of label to ``{"latent": <path>, "model": <path>}``.
    """
    print("Loading spring transformer model...")
    model = torch.jit.load(model_path, map_location=torch.device("cpu"))

    output_paths = {}
    for label, mcap_file_paths in datasets.items():
        print(f"Computing frozen latent for '{label}' ({len(mcap_file_paths)} MCAPs)...")
        frozen_latent, num_windows = compute_frozen_latent(model, mcap_file_paths)

        # Latent payload for the frozen-latent mode of the full checkpoint.
        payload = {
            "latent": frozen_latent,
            "input_columns": model.input_columns,
            "num_windows": num_windows,
            "checkpoint": model_path,
        }
        latent_path = f"{OUTPUT_DIR}frozen_latent_{label}.pt"
        torch.save(payload, latent_path)
        print(f"  wrote {latent_path} ({num_windows} windows)")

        # Pruned deployment-only scripted model with the latent embedded.
        pruned_model = build_frozen_latent_model(model, frozen_latent)
        wrapped_model = ScaledModelWrapper(
            pruned_model,
            model.input_mean,
            model.input_std,
            model.output_mean,
            model.output_std,
            frequency=model.metadata["frequency"],
            history_size=model.metadata["history_size"],
            stride=model.metadata["stride"],
            input_columns=model.input_columns,
            output_columns=model.output_columns,
        )
        wrapped_model.freeze()
        model_output_path = f"{OUTPUT_DIR}spring_transformer_frozen_{label}.pt"
        wrapped_model.script_and_save(model_output_path)
        print(f"  wrote {model_output_path}")

        output_paths[label] = {"latent": latent_path, "model": model_output_path}

    return output_paths


def main():
    export_frozen_deployment(DEFAULT_MODEL_PATH, INPUT_DATASETS)


if __name__ == "__main__":
    main()
