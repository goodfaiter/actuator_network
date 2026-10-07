"""Fit the current-to-torque gain (P gain) from powered wind/unwind MCAP recordings.

With the motor driving a spring through a pulley and tendon, the bidirectional
constant-velocity sweep cancels friction:

    wind   (+v): K_t * i_+ = tau_ext + tau_f(v)
    unwind (-v): K_t * i_- = tau_ext - tau_f(v)

Wind and unwind plateaus at the same |velocity| are paired, binned by measured
position, and each bin contributes the mean of the two currents and the mean
load torque, giving

    K_t * 0.5 * (i_+ + i_-) = tau_ext

so regressing the load torque against the mean current yields K_t, the
effective torque per amp at the output. The half current difference gives the
friction, tau_f(v) = K_t * 0.5 * (i_+ - i_-), reported against the back-drive
friction model b*v + c*sign(v); with a load-dependent gearbox the powered
friction is typically much larger than the back-drive constants.

Only velocity plateaus are used: a sample belongs to a plateau when it lies in
a contiguous run of identical desired velocity with |desired velocity| >=
VELOCITY_THRESHOLD and at least MIN_PASS_SAMPLES moving samples (measured
|v| >= VELOCITY_THRESHOLD), so the velocity-command ramps (acceleration parts)
and the zero-velocity reversal dwell are excluded by construction.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from actuator_network.helpers.data_pipeline import load_mcap_dataframes_parallel_cached
from actuator_network.helpers.pandas_processing import process_dataframe

OUTPUT_DIR = "/workspace/data/output_data/"
DATA_FREQ = 200

CURRENT_COL = "measured_current_amp_data"
DESIRED_VELOCITY_COL = "desired_velocity_rad_per_sec_data"
VELOCITY_COL = "measured_velocity_rad_per_sec_data"
POSITION_COL = "measured_position_rad_data"
TAU_COL = "bota_wrench_N_and_Nm_torque_z"

VELOCITY_THRESHOLD = 0.05
MIN_PASS_SAMPLES = 100
NUM_BINS = 20
MIN_BIN_SAMPLES = 5

INERTIA_PARAMS_PATH = "/workspace/data/output_data/inertia_params.json"
DEFAULT_FRICTION_PARAMS = {"b": 0.003764099167855621, "c": 0.006662410948835853}


def load_friction_params(params_path: str = INERTIA_PARAMS_PATH) -> dict:
    """Load the back-drive friction constants b and c, falling back to defaults."""
    if os.path.isfile(params_path):
        with open(params_path) as f:
            params = json.load(f)
        return {"b": float(params["b"]), "c": float(params["c"])}
    return dict(DEFAULT_FRICTION_PARAMS)


def segment_plateaus(
    desired_velocity: np.ndarray, measured_velocity: np.ndarray, min_samples: int, velocity_threshold: float
) -> list[tuple[float, np.ndarray]]:
    """Find velocity plateaus as contiguous runs of identical desired velocity.

    A plateau is kept when |desired velocity| >= ``velocity_threshold`` and at
    least ``min_samples`` of its samples are actually moving (measured |v| >=
    ``velocity_threshold``). The returned mask is restricted to the moving
    samples, so samples on the velocity-command ramps never enter the fit.

    Args:
        desired_velocity: Commanded velocity signal (rad/s), piecewise constant.
        measured_velocity: Measured velocity signal (rad/s).
        min_samples: Minimum number of moving samples required for a plateau.
        velocity_threshold: |velocity| below which samples count as static.

    Returns:
        List of (level, mask) pairs, in order of appearance.
    """
    boundaries = np.concatenate([[True], desired_velocity[1:] != desired_velocity[:-1]])
    run_id = np.cumsum(boundaries)
    plateaus = []
    for rid in range(1, run_id.max() + 1):
        run_mask = run_id == rid
        level = float(desired_velocity[run_mask][0])
        if abs(level) < velocity_threshold:
            continue
        moving = run_mask & (np.abs(measured_velocity) >= velocity_threshold)
        if moving.sum() < min_samples:
            continue
        plateaus.append((level, moving))
    return plateaus


def build_gain_points(dataframes: list[pd.DataFrame], velocity_threshold: float, min_samples: int, num_bins: int) -> dict[str, np.ndarray]:
    """Pair wind/unwind plateaus and build bin-averaged gain points.

    Wind (+v) and unwind (-v) plateaus at the same |level| are paired. Samples
    are binned by measured position over the overlapping range; each bin
    contributes the mean of the two pass currents, the mean load torque
    (friction cancels) and the half current difference (friction estimate).

    Args:
        dataframes: Processed DataFrames (output of ``process_dataframe``).
        velocity_threshold: |velocity| below which samples count as static.
        min_samples: Minimum plateau length in samples.
        num_bins: Number of position bins per plateau pair.

    Returns:
        Dict with concatenated ``level``, ``i_bar``, ``tau_ext`` and
        ``i_half_diff`` arrays.
    """
    required = (CURRENT_COL, DESIRED_VELOCITY_COL, VELOCITY_COL, POSITION_COL, TAU_COL)
    chunks: dict[str, list[float]] = {name: [] for name in ("level", "i_bar", "tau_ext", "i_half_diff")}
    for df in dataframes:
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"  skipping recording missing columns: {missing}")
            continue
        current = df[CURRENT_COL].to_numpy(dtype=np.float64)
        tau = df[TAU_COL].to_numpy(dtype=np.float64)
        desired_velocity = df[DESIRED_VELOCITY_COL].to_numpy(dtype=np.float64)
        measured_velocity = df[VELOCITY_COL].to_numpy(dtype=np.float64)
        position = df[POSITION_COL].to_numpy(dtype=np.float64)

        plateaus = segment_plateaus(desired_velocity, measured_velocity, min_samples=min_samples, velocity_threshold=velocity_threshold)
        by_level = {round(level, 4): mask for level, mask in plateaus}
        for level in sorted({abs(key) for key in by_level}):
            mask_wind = by_level.get(round(level, 4))
            mask_unwind = by_level.get(round(-level, 4))
            if mask_wind is None or mask_unwind is None:
                continue
            lo = max(np.nanmin(position[mask_wind]), np.nanmin(position[mask_unwind]))
            hi = min(np.nanmax(position[mask_wind]), np.nanmax(position[mask_unwind]))
            if not hi > lo:
                continue
            bin_idx = np.digitize(position, np.linspace(lo, hi, num_bins + 1)) - 1
            for bin_number in range(num_bins):
                wind_bin = mask_wind & (bin_idx == bin_number)
                unwind_bin = mask_unwind & (bin_idx == bin_number)
                if wind_bin.sum() < MIN_BIN_SAMPLES or unwind_bin.sum() < MIN_BIN_SAMPLES:
                    continue
                i_wind = float(current[wind_bin].mean())
                i_unwind = float(current[unwind_bin].mean())
                chunks["level"].append(level)
                chunks["i_bar"].append(0.5 * (i_wind + i_unwind))
                chunks["tau_ext"].append(0.5 * (-tau[wind_bin].mean() - tau[unwind_bin].mean()))
                chunks["i_half_diff"].append(0.5 * (i_wind - i_unwind))
    return {name: np.array(values) for name, values in chunks.items()}


def _fit_slope(current: np.ndarray, tau_ext: np.ndarray) -> float:
    """Least-squares slope of tau_ext against current through the origin."""
    k, _, _, _ = np.linalg.lstsq(current[:, None], tau_ext, rcond=None)
    return float(k[0])


def _metrics(tau_ext: np.ndarray, current: np.ndarray, k: float) -> dict[str, float]:
    """RMSE and R^2 of tau_ext against k * current."""
    residual = tau_ext - k * current
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((tau_ext - np.mean(tau_ext)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan")
    return {"rmse": rmse, "r2": r2}


def fit_gain(points: dict[str, np.ndarray]) -> dict:
    """Fit K_t as the slope of the load torque against the mean current.

    The load torque is ``-torque_z``; if the fitted slope is negative the sign
    convention is flipped and the fit is repeated (K_t must be positive).

    Args:
        points: Output of :func:`build_gain_points`.

    Returns:
        Dict with ``K_t``, ``sign_flipped``, ``rmse`` and ``r2``.
    """
    k = _fit_slope(points["i_bar"], points["tau_ext"])
    sign_flipped = bool(k < 0)
    if sign_flipped:
        tau_ext = -points["tau_ext"]
        k = _fit_slope(points["i_bar"], tau_ext)
    else:
        tau_ext = points["tau_ext"]
    return {"K_t": k, "sign_flipped": sign_flipped, **_metrics(tau_ext, points["i_bar"], k)}


def per_level_gains(points: dict[str, np.ndarray], fit: dict, friction: dict) -> dict[float, dict]:
    """Per-|level| gain, fit quality and recovered friction.

    The recovered friction is ``tau_f = K_t(level) * mean(i_half_diff)`` at
    that level, reported against the back-drive model ``b*v + c``.

    Args:
        points: Output of :func:`build_gain_points`.
        fit: Output of :func:`fit_gain` (provides the sign convention).
        friction: Back-drive friction constants with keys ``b`` and ``c``.

    Returns:
        Dict keyed by |level| with ``K_t``, ``rmse``, ``r2``, ``num_points``,
        ``tau_f`` and ``tau_f_backdrive_model``.
    """
    tau_ext = -points["tau_ext"] if fit["sign_flipped"] else points["tau_ext"]
    results = {}
    for level in sorted(set(points["level"].tolist())):
        sel = points["level"] == level
        k = _fit_slope(points["i_bar"][sel], tau_ext[sel])
        results[float(level)] = {
            "K_t": k,
            **_metrics(tau_ext[sel], points["i_bar"][sel], k),
            "num_points": int(sel.sum()),
            "tau_f": float(k * points["i_half_diff"][sel].mean()),
            "tau_f_backdrive_model": float(friction["b"] * level + friction["c"]),
        }
    return results


def plot_gain_fit(points: dict[str, np.ndarray], fit: dict, level_gains: dict[float, dict], friction: dict, output_path: str) -> None:
    """Plot the gain curve, recovered friction and the gain regression."""
    tau_ext = -points["tau_ext"] if fit["sign_flipped"] else points["tau_ext"]
    levels = sorted(level_gains)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), dpi=150)

    ax1.plot(levels, [level_gains[level]["K_t"] for level in levels], "o-", label="Per level")
    ax1.axhline(fit["K_t"], color="red", linestyle="--", label=f"Global = {fit['K_t']:.3f}")
    ax1.set_xlabel("|velocity| [rad/s]")
    ax1.set_ylabel("K_t [Nm/A]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(levels, [level_gains[level]["tau_f"] for level in levels], "o-", label="Recovered (powered)")
    model_levels = np.linspace(0.0, max(levels), 100)
    ax2.plot(model_levels, friction["b"] * model_levels + friction["c"], label="Back-drive b*v+c")
    ax2.set_xlabel("|velocity| [rad/s]")
    ax2.set_ylabel("tau_f [Nm]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    scatter = ax3.scatter(points["i_bar"], tau_ext, c=points["level"], cmap="viridis", s=10)
    fig.colorbar(scatter, ax=ax3, label="|velocity| [rad/s]")
    i_max = float(points["i_bar"].max())
    ax3.plot([0.0, i_max], [0.0, fit["K_t"] * i_max], "r--", label=f"K_t = {fit['K_t']:.3f}")
    ax3.set_xlabel("Mean current [A]")
    ax3.set_ylabel("Load torque [Nm]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, fit: dict, level_gains: dict[float, dict], output_path: str, data_freq: int = DATA_FREQ) -> None:
    """Overlay measured and predicted load torque over the longest plateau."""
    current = df[CURRENT_COL].to_numpy(dtype=np.float64)
    tau = df[TAU_COL].to_numpy(dtype=np.float64)
    desired_velocity = df[DESIRED_VELOCITY_COL].to_numpy(dtype=np.float64)
    measured_velocity = df[VELOCITY_COL].to_numpy(dtype=np.float64)
    plateaus = segment_plateaus(desired_velocity, measured_velocity, MIN_PASS_SAMPLES, VELOCITY_THRESHOLD)
    if not plateaus:
        return
    level, mask = max(plateaus, key=lambda item: int(item[1].sum()))
    tau_ext = (-tau if not fit["sign_flipped"] else tau)[mask]
    prediction = fit["K_t"] * current[mask] - level_gains[round(abs(level), 4)]["tau_f"]
    time = np.arange(int(mask.sum())) / data_freq

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(time, tau_ext, label="Measured", alpha=0.7)
    ax.plot(time, prediction, label="Predicted")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Load torque [Nm]")
    ax.set_title(f"Plateau at {level:+.2f} rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_10_07/rosbag2_2026_10_07-13_43_09_0.mcap",
    ]
    val_mcap_file_paths = []

    if not mcap_file_paths:
        print("No MCAP files provided; add powered wind/unwind recordings to mcap_file_paths in train_current_gain.py.")
        return

    print("Loading and processing MCAP files...")
    train_dataframes = load_mcap_dataframes_parallel_cached(mcap_file_paths, freq=DATA_FREQ)
    val_dataframes = load_mcap_dataframes_parallel_cached(val_mcap_file_paths, freq=DATA_FREQ) if val_mcap_file_paths else []
    for df in train_dataframes + val_dataframes:
        process_dataframe(df)

    friction = load_friction_params()

    train_points = build_gain_points(
        train_dataframes, velocity_threshold=VELOCITY_THRESHOLD, min_samples=MIN_PASS_SAMPLES, num_bins=NUM_BINS
    )
    print(f"Using {len(train_points['tau_ext'])} bin-averaged points from velocity plateaus")
    if len(train_points["tau_ext"]) == 0:
        print("No plateau points found; cannot fit.")
        return

    fit = fit_gain(train_points)
    print("\nGlobal fit (tau_ext = K_t * mean current, friction cancels):")
    print(f"  K_t = {fit['K_t']:.6g} Nm/A")
    if fit["sign_flipped"]:
        print("  load torque sign flipped (fitted slope was negative)")
    print(f"  RMSE: {fit['rmse']:.6g} Nm, R^2: {fit['r2']:.4f}")

    level_gains = per_level_gains(train_points, fit, friction)
    print("\nPer-level gains (gain curve):")
    for level, result in level_gains.items():
        print(
            f"  |v| = {level:.2f} rad/s: K_t = {result['K_t']:+.4f} Nm/A, R^2 = {result['r2']:+.3f}, "
            f"tau_f = {result['tau_f']:+.5f} Nm (back-drive model {result['tau_f_backdrive_model']:+.5f} Nm)"
        )

    val_results = {}
    if val_dataframes:
        val_points = build_gain_points(
            val_dataframes, velocity_threshold=VELOCITY_THRESHOLD, min_samples=MIN_PASS_SAMPLES, num_bins=NUM_BINS
        )
        if len(val_points["tau_ext"]) > 0:
            tau_ext = -val_points["tau_ext"] if fit["sign_flipped"] else val_points["tau_ext"]
            val_results = _metrics(tau_ext, val_points["i_bar"], fit["K_t"])
            val_results["num_points"] = int(len(val_points["tau_ext"]))
            print(f"\nValidation: RMSE {val_results['rmse']:.6g} Nm, R^2 {val_results['r2']:.4f} ({val_results['num_points']} points)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "K_t": fit["K_t"],
        "sign_flipped": fit["sign_flipped"],
        "velocity_threshold_rad_per_sec": VELOCITY_THRESHOLD,
        "min_pass_samples": MIN_PASS_SAMPLES,
        "num_bins": NUM_BINS,
        "friction_params": friction,
        "train": {
            "num_points": int(len(train_points["tau_ext"])),
            "rmse": fit["rmse"],
            "r2": fit["r2"],
            "per_level": {f"{level:.2f}": result for level, result in level_gains.items()},
        },
        "val": val_results,
    }
    params_path = os.path.join(OUTPUT_DIR, "current_gain_params.json")
    with open(params_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {params_path}")

    plot_gain_fit(train_points, fit, level_gains, friction, os.path.join(OUTPUT_DIR, "current_gain_fit.png"))
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'current_gain_fit.png')}")
    plot_timeseries(train_dataframes[0], fit, level_gains, os.path.join(OUTPUT_DIR, "current_gain_timeseries.png"))
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'current_gain_timeseries.png')}")


if __name__ == "__main__":
    main()
