"""Fit the M5 extended friction model to the friction envelope (max friction).

    tau_f^m = K_v |v| + K_c + |K_m tau_m - K_e tau_e|
            + exp(-|v / v_s|^alpha) * (K_c^s + |K_m^s tau_m - K_e^s tau_e|)

The measured friction is observed per sample from the fixed constants fitted in
separate experiments (see ``train_inertia`` and ``train_current_gain``):

    tau_f = tau_m - J * alpha + tau_e,   tau_m = K_t * i_des,
    i_des = POS_K * (theta_desired - theta)

where ``i_des`` is the P-control commanded current and ``J``, ``K_t`` and
``POS_K`` are the fixed inertia, current-to-torque gain and position P gain.
The velocity-proportional and Coulomb terms are fixed to the back-drive
constants (``K_v = b``, ``K_c = c``), so only the load-dependent and
static-envelope parameters are optimized. The envelope is only observable in
some samples:

- Moving (|v| > ``velocity_threshold``): friction is saturated and opposes the
  motion, so ``tau_f^m = -sign(v) * tau_f``. Fitted with MSE.
- Breakaway (last static sample before motion starts, after at least
  ``breakaway_min_static_samples`` static samples): friction has just reached its
  maximum, so ``tau_f^m = max |tau_f|`` over the last ``breakaway_window`` static
  samples. Fitted with MSE.
- Other static samples (|v| <= ``static_velocity_threshold``): friction only
  needs to hold the joint, so ``|tau_f| <= tau_f^m``. Enforced with a one-sided
  hinge penalty, plus a weaker penalty on the slack so the envelope stays tight.
- Samples between the two velocity thresholds, or in moving runs shorter than
  ``min_moving_samples``, are ambiguous and excluded.

The model applies a velocity dead zone of ``static_velocity_threshold`` so rest
samples are evaluated at zero velocity during training and inference alike.

Hyperparameters are read from ``wandb.config`` so this script can be used as the
program for a W&B sweep agent. When run manually, it falls back to the defaults.
"""

import json
import os

import numpy as np
import pandas as pd
import torch

import wandb
from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel_cached
from actuator_network.helpers.hyperparameters import M5FrictionConfig
from actuator_network.helpers.pandas_processing import process_dataframe
from actuator_network.helpers.torch_model import M5EnvelopeFrictionModel

OUTPUT_DIR = "/workspace/data/output_data/"
DEFAULT_WANDB_PROJECT = "actuator_network"

VELOCITY_COL = "measured_velocity_rad_per_sec_data"
TAU_MOTOR_COL = "calculated_motor_torque_Nm_data"
TAU_EXTERNAL_COL = "bota_wrench_N_and_Nm_torque_z"
FRICTION_COL = "calculated_friction_torque_Nm_data"

FIXED_K_V = 0.003764099167855621
FIXED_K_C = 0.006662410948835853


def _run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the position within, and the total length of, each sample's constant run of ``mask``."""
    run_id = np.cumsum(np.concatenate([[True], mask[1:] != mask[:-1]]))
    groups = pd.Series(run_id).groupby(run_id)
    return groups.cumcount().to_numpy() + 1, groups.transform("size").to_numpy()


def build_friction_samples(
    dataframes: list[pd.DataFrame],
    velocity_threshold: float,
    breakaway_min_static_samples: int,
    device: torch.device,
    static_velocity_threshold: float | None = None,
    min_moving_samples: int = 1,
    breakaway_window: int = 1,
) -> dict[str, torch.Tensor]:
    """Extract model inputs, envelope targets and sample-type masks.

    Breakaway points are detected per DataFrame so that runs never cross file
    boundaries. Samples in the dead band between ``static_velocity_threshold``
    and ``velocity_threshold`` (or in too-short moving runs) are ambiguous and
    dropped, as are non-finite rows, after detection.

    Args:
        dataframes: Processed DataFrames (output of ``process_dataframe``).
        velocity_threshold: |velocity| above which a sample counts as moving.
        breakaway_min_static_samples: Minimum static run length before a breakaway.
        device: Torch device for the returned tensors.
        static_velocity_threshold: |velocity| at or below which a sample counts as
            static. Defaults to ``velocity_threshold`` (no dead band).
        min_moving_samples: Minimum moving run length for moving samples.
        breakaway_window: Number of static samples up to the breakaway point over
            which the maximum |tau_f| is taken as the breakaway target.

    Returns:
        Dict with ``velocity``, ``tau_motor``, ``tau_external``, ``target`` and
        boolean masks ``moving``, ``breakaway`` and ``static``. ``target`` is
        ``-sign(v) * tau_f`` for moving samples, the windowed max of ``|tau_f|``
        for breakaway samples and ``|tau_f|`` for static samples.
    """
    if static_velocity_threshold is None:
        static_velocity_threshold = velocity_threshold

    keys = ("velocity", "tau_motor", "tau_external", "target", "moving", "breakaway", "static")
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    for df in dataframes:
        velocity = df[VELOCITY_COL].to_numpy(dtype=np.float64)
        friction = df[FRICTION_COL].to_numpy(dtype=np.float64)
        speed = np.abs(velocity)
        static = speed <= static_velocity_threshold
        moving = speed > velocity_threshold
        moving &= _run_lengths(moving)[1] >= min_moving_samples

        # A breakaway is the last sample of a long enough static run whose next
        # classified (static or moving) sample is moving; dead-band samples in
        # between are skipped.
        state = pd.Series(np.where(moving, 1.0, np.where(static, 0.0, np.nan)))
        next_state = state.shift(-1).bfill().to_numpy()
        breakaway = static & (next_state == 1.0) & (_run_lengths(static)[0] >= breakaway_min_static_samples)

        # NaNs (non-static samples) are skipped by the rolling max.
        static_abs_friction = pd.Series(np.where(static, np.abs(friction), np.nan))
        breakaway_target = static_abs_friction.rolling(breakaway_window, min_periods=1).max().to_numpy()

        chunks["velocity"].append(velocity)
        chunks["tau_motor"].append(df[TAU_MOTOR_COL].to_numpy(dtype=np.float64))
        chunks["tau_external"].append(df[TAU_EXTERNAL_COL].to_numpy(dtype=np.float64))
        chunks["target"].append(np.where(moving, -np.sign(velocity) * friction, np.where(breakaway, breakaway_target, np.abs(friction))))
        chunks["moving"].append(moving)
        chunks["breakaway"].append(breakaway)
        chunks["static"].append(static & ~breakaway)

    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    keep = arrays["moving"] | arrays["breakaway"] | arrays["static"]
    keep &= np.isfinite(arrays["velocity"]) & np.isfinite(arrays["tau_motor"]) & np.isfinite(arrays["tau_external"])
    keep &= np.isfinite(arrays["target"])

    samples = {}
    for k, v in arrays.items():
        dtype = torch.bool if v.dtype == bool else torch.float32
        samples[k] = torch.as_tensor(v[keep], dtype=dtype, device=device)
    return samples


def friction_sample_kwargs(config: M5FrictionConfig, device: torch.device) -> dict:
    """Keyword arguments for ``build_friction_samples`` taken from ``config``."""
    return {
        "velocity_threshold": config.velocity_threshold,
        "breakaway_min_static_samples": config.breakaway_min_static_samples,
        "device": device,
        "static_velocity_threshold": config.static_velocity_threshold,
        "min_moving_samples": config.min_moving_samples,
        "breakaway_window": config.breakaway_window,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over ``mask``; zero if the mask is empty."""
    if not mask.any():
        return values.new_zeros(())
    return values[mask].mean()


def envelope_losses(
    model: M5EnvelopeFrictionModel,
    samples: dict[str, torch.Tensor],
    config: M5FrictionConfig,
) -> dict[str, torch.Tensor]:
    """Compute the individual loss terms and their weighted total."""
    prediction = model(samples["velocity"], samples["tau_motor"], samples["tau_external"])
    error_sq = (prediction - samples["target"]) ** 2
    # Static samples: strongly penalize held friction above the envelope, weakly
    # penalize envelope slack so the envelope stays tight (asymmetric least squares).
    violation_sq = torch.relu(samples["target"] - prediction) ** 2
    slack_sq = torch.relu(prediction - samples["target"]) ** 2

    losses = {
        "moving_mse": _masked_mean(error_sq, samples["moving"]),
        "breakaway_mse": _masked_mean(error_sq, samples["breakaway"]),
        "static_bound": _masked_mean(violation_sq, samples["static"]),
        "static_tightness": _masked_mean(slack_sq, samples["static"]),
    }
    losses["total"] = (
        losses["moving_mse"]
        + config.breakaway_weight * losses["breakaway_mse"]
        + config.static_bound_weight * losses["static_bound"]
        + config.static_tightness_weight * losses["static_tightness"]
    )
    return losses


def train_m5(
    config: M5FrictionConfig,
    train_samples: dict[str, torch.Tensor],
    val_samples: dict[str, torch.Tensor],
    device: torch.device,
    latest_prefix: str = "m5_",
) -> M5EnvelopeFrictionModel:
    """Fit the M5 envelope model with full-batch Adam and early stopping on validation loss.

    Args:
        config: Hyperparameter configuration.
        train_samples: Output of ``build_friction_samples`` for the training set.
        val_samples: Output of ``build_friction_samples`` for the validation set.
        device: Torch device.
        latest_prefix: Filename prefix for the saved parameters.

    Returns:
        The fitted model with the best validation parameters loaded.
    """
    for name, samples in (("train", train_samples), ("val", val_samples)):
        print(
            f"  {name}: {samples['moving'].numel()} samples, {int(samples['moving'].sum())} moving, "
            f"{int(samples['breakaway'].sum())} breakaway, {int(samples['static'].sum())} static"
        )

    model = M5EnvelopeFrictionModel(
        device=device,
        velocity_deadzone=config.static_velocity_threshold,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        fixed_params={"K_v": FIXED_K_V, "K_c": FIXED_K_C},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(config.patience // 4, 1))

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        train_losses = envelope_losses(model, train_samples, config)
        train_losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = envelope_losses(model, val_samples, config)
        val_loss = val_losses["total"].item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 100 == 0 or epoch == config.num_epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}], Train Loss: {train_losses['total'].item():.3e}, "
                f"Val Loss: {val_loss:.3e}, LR: {current_lr:.6f}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "learning_rate": current_lr,
                    **{f"train_{k}": v.item() for k, v in train_losses.items()},
                    **{f"val_{k}": v.item() for k, v in val_losses.items()},
                    **{f"param/{k}": v for k, v in model.physical_parameters().items()},
                }
            )

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {config.patience} epochs).")
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        val_losses = {k: v.item() for k, v in envelope_losses(model, val_samples, config).items()}
    params = model.physical_parameters()

    print("\nFitted parameters:")
    print(f"  fixed: {', '.join(model.fixed_names)}")
    for name, value in params.items():
        print(f"  {name} = {value:.6g}")
    print(f"Best val loss: {val_losses['total']:.3e}")
    print(f"Val moving RMSE: {val_losses['moving_mse'] ** 0.5:.6f} Nm, breakaway RMSE: {val_losses['breakaway_mse'] ** 0.5:.6f} Nm")

    wandb.run.summary.update(
        {
            "best_val_loss": val_losses["total"],
            "val_moving_rmse": val_losses["moving_mse"] ** 0.5,
            "val_breakaway_rmse": val_losses["breakaway_mse"] ** 0.5,
            "val_static_bound": val_losses["static_bound"],
            "val_static_tightness": val_losses["static_tightness"],
            **{f"param/{k}": v for k, v in params.items()},
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f"{latest_prefix}model.pt")
    params_path = os.path.join(OUTPUT_DIR, f"{latest_prefix}params.json")
    torch.save(model.state_dict(), model_path)
    with open(params_path, "w") as f:
        json.dump({**params, "fixed": list(model.fixed_names)}, f, indent=2)
    print(f"Saved model to {model_path} and parameters to {params_path}")

    return model


def main():
    if wandb.run is None:
        wandb.init(project=DEFAULT_WANDB_PROJECT)

    config = M5FrictionConfig.from_wandb_config(wandb.config)
    print(f"Using configuration: {config}")

    if not config.is_valid():
        print(f"Skipping invalid configuration: {config}")
        wandb.run.summary["skipped_invalid"] = True
        wandb.finish()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcap_file_paths = [
        # blocked
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_01_21_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_03_39_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_05_41_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_13_54_0.mcap",
        # strong spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_38_20_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_39_52_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_42_57_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_44_39_0.mcap",
        # weak spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_18_11_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_20_54_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_23_21_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_24_47_0.mcap",
        # finger
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_11_54_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_13_57_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_15_28_0.mcap",
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_17_53_0.mcap",
    ]
    val_mcap_file_paths = [
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-09_16_16_0.mcap",  # blocked
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-10_46_40_0.mcap",  # strong spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-11_27_33_0.mcap",  # weak spring
        "/workspace/data/training_data/2026_09_16/rosbag2_2026_09_16-12_19_09_0.mcap",  # finger
    ]

    print("Loading and processing MCAP files...")
    train_dataframes = load_mcap_dataframes_parallel_cached(mcap_file_paths, freq=config.data_freq)
    val_dataframes = load_mcap_dataframes_parallel_cached(val_mcap_file_paths, freq=config.data_freq)
    # The parquet cache is not keyed on the processing code; recompute derived torque columns.
    for df in train_dataframes + val_dataframes:
        process_dataframe(df)

    sample_kwargs = friction_sample_kwargs(config, device)
    train_samples = build_friction_samples(train_dataframes, **sample_kwargs)
    val_samples = build_friction_samples(val_dataframes, **sample_kwargs)

    latest_prefix = f"m5_sweep_{wandb.run.id}_" if wandb.run.sweep_id is not None else "m5_"

    print("Running training...")
    train_m5(config, train_samples, val_samples, device, latest_prefix=latest_prefix)


if __name__ == "__main__":
    main()
