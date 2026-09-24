"""Run deployment inference with the pruned frozen-latent spring transformer.

The pruned deployment checkpoints (``spring_transformer_frozen_<label>.pt``,
exported by ``actuator_network.export_frozen_latent``) contain strictly the
deployed components — the compiled force transformer, the spring coefficient
head, the stateful force buffer, and an embedded frozen latent computed from a
dataset of one specific spring coefficient. The model transformer never runs at
deployment: the force transformer receives the embedded latent every tick, so
forces are permanently pinned to that specific spring coefficient, and the
script loads strictly less memory than the full adaptive checkpoint.

Usage:

.. code-block:: bash

    uv run test-spring-transformer-frozen

"""

from actuator_network.test_spring_transformer import run_spring_transformer_inference

# One pruned checkpoint per exported spring-coefficient label (see
# `export-frozen-latent`); swap the label to deploy a different spring.
DEFAULT_MODEL_PATH = "/workspace/data/output_data/spring_transformer_frozen_finger.pt"

OUTPUT_SUFFIX = "_spring_transformer_frozen_predicted"


def run_frozen_transformer_inference(
    model_path: str,
    mcap_file_paths: list[str],
) -> list[str]:
    """Run frozen-latent deployment inference on the given MCAPs.

    One model call is one force tick, so inference is performed at the model's
    inference rate (every ``stride``-th preprocessed sample) by feeding the
    latest normalized sample; the force buffer advances every tick while the
    embedded frozen latent stays constant. A fresh reset is performed per
    recording.

    Args:
        model_path: Path to the pruned frozen-latent deployment checkpoint.
        mcap_file_paths: List of input MCAP files.

    Returns:
        List of output file paths (suffix ``_spring_transformer_frozen_predicted``).
    """
    return run_spring_transformer_inference(
        model_path,
        mcap_file_paths,
        output_suffix=OUTPUT_SUFFIX,
        require_pruned=True,
    )


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_16_16_0.mcap",  # blocked
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_46_40_0.mcap",  # strong spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_27_33_0.mcap",  # weak spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_19_09_0.mcap",  # finger
    ]

    run_frozen_transformer_inference(DEFAULT_MODEL_PATH, mcap_file_paths)


if __name__ == "__main__":
    main()
