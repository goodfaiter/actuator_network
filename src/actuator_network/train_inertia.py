"""Fit the motor inertia from back-drive (current = 0) MCAP recordings.

With the motor passive, the tendon torque measured by the BOTA sensor balances
the reflected motor dynamics:

    tau = J*alpha + b*omega + c*sign(omega)

where tau is ``bota_wrench_N_and_Nm_torque_z``, omega is
``measured_velocity_rad_per_sec_data`` and alpha is the angular acceleration
(``calculated_acceleration_rad_per_sec2_data``, the derivative of the measured
velocity). Every sample contributes one row ``[alpha, omega, sign(omega)]`` to
a regressor matrix that is solved for ``[J, b, c]`` with ordinary least
squares.

Samples with |omega| below ``VELOCITY_THRESHOLD`` are dropped so the fit stays
above the static/stick-slip (Stribeck) region, where friction is
load-dependent and the sensor torque is not a clean function of speed. The
fitted J is the output-side reflected inertia; with a load-dependent gearbox
it is effectively J/eta_back for back-driving, and c is an effective Coulomb
level above the Stribeck region. If the fitted J comes out negative the
sensor torque axis convention is assumed flipped and tau is negated before
refitting, since a passive motor must have positive inertia and damping.
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

VELOCITY_COL = "measured_velocity_rad_per_sec_data"
ACCELERATION_COL = "calculated_acceleration_rad_per_sec2_data"
TAU_COL = "bota_wrench_N_and_Nm_torque_z"
VELOCITY_THRESHOLD = 0.05


def build_regression_samples(dataframes: list[pd.DataFrame], velocity_threshold: float = VELOCITY_THRESHOLD) -> dict[str, np.ndarray]:
    """Extract regression rows (alpha, omega, tau) from processed DataFrames.

    The first sample of each recording is dropped because its derivative is a
    fill artifact. Rows with |omega| below ``velocity_threshold`` or non-finite
    values are excluded so the fit stays above the static/stick-slip region.

    Args:
        dataframes: Processed DataFrames (output of ``process_dataframe``).
        velocity_threshold: |omega| below which a sample is dropped.

    Returns:
        Dict with concatenated ``alpha``, ``omega`` and ``tau`` arrays.
    """
    chunks: dict[str, list[np.ndarray]] = {"alpha": [], "omega": [], "tau": []}
    for df in dataframes:
        omega = df[VELOCITY_COL].to_numpy(dtype=np.float64)
        alpha = df[ACCELERATION_COL].to_numpy(dtype=np.float64)
        tau = df[TAU_COL].to_numpy(dtype=np.float64)
        omega, alpha, tau = omega[1:], alpha[1:], tau[1:]
        keep = (np.abs(omega) >= velocity_threshold) & np.isfinite(omega) & np.isfinite(alpha) & np.isfinite(tau)
        chunks["alpha"].append(alpha[keep])
        chunks["omega"].append(omega[keep])
        chunks["tau"].append(tau[keep])
    return {name: np.concatenate(values) for name, values in chunks.items()}


def _regressor(alpha: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Build the regressor matrix [alpha, omega, sign(omega)] for [J, b, c]."""
    return np.column_stack([alpha, omega, np.sign(omega)])


def _solve(alpha: np.ndarray, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Solve the least-squares fit for [J, b, c]."""
    coefficients, _, _, _ = np.linalg.lstsq(_regressor(alpha, omega), tau, rcond=None)
    return coefficients


def evaluate(params: dict, samples: dict[str, np.ndarray]) -> dict[str, float]:
    """Score fitted [J, b, c] against regression samples.

    Args:
        params: Dict with ``J``, ``b``, ``c`` and ``sign_flipped``.
        samples: Dict with ``alpha``, ``omega`` and ``tau`` arrays.

    Returns:
        Dict with ``rmse`` and ``r2`` of tau against J*alpha + b*omega +
        c*sign(omega).
    """
    tau = -samples["tau"] if params["sign_flipped"] else samples["tau"]
    fitted = params["J"] * samples["alpha"] + params["b"] * samples["omega"] + params["c"] * np.sign(samples["omega"])
    residual = tau - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((tau - np.mean(tau)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan")
    return {"rmse": rmse, "r2": r2}


def fit_inertia(alpha: np.ndarray, omega: np.ndarray, tau: np.ndarray) -> dict:
    """Fit tau = J*alpha + b*omega + c*sign(omega) with ordinary least squares.

    If the fitted J is negative, the sensor torque axis convention is assumed
    flipped and tau is negated before refitting, since a passive motor must
    have positive inertia and damping.

    Args:
        alpha: Angular acceleration samples (rad/s^2).
        omega: Angular velocity samples (rad/s).
        tau: Measured torque samples (Nm).

    Returns:
        Dict with ``J`` [Nm*s^2/rad], ``b`` [Nm*s/rad], ``c`` [Nm] and
        ``sign_flipped``, plus the diagnostics from :func:`evaluate` and
        ``condition_number`` of the regressor, ``alpha_omega_correlation``
        (identifiability of b) and ``inertia_torque_share`` (mean |J*alpha|
        over mean |tau|).
    """
    coefficients = _solve(alpha, omega, tau)
    sign_flipped = bool(coefficients[0] < 0)
    if sign_flipped:
        coefficients = _solve(alpha, omega, -tau)
    params = {"J": float(coefficients[0]), "b": float(coefficients[1]), "c": float(coefficients[2]), "sign_flipped": sign_flipped}

    std_alpha = float(np.std(alpha))
    std_omega = float(np.std(omega))
    mean_abs_tau = float(np.mean(np.abs(tau)))
    diagnostics = {
        **evaluate(params, {"alpha": alpha, "omega": omega, "tau": tau}),
        "condition_number": float(np.linalg.cond(_regressor(alpha, omega))),
        "alpha_omega_correlation": float(np.corrcoef(alpha, omega)[0, 1]) if std_alpha > 0 and std_omega > 0 else float("nan"),
        "inertia_torque_share": float(np.mean(np.abs(params["J"] * alpha)) / mean_abs_tau) if mean_abs_tau > 0 else float("nan"),
    }
    return {**params, **diagnostics}


def plot_tau_fit(df: pd.DataFrame, params: dict, output_path: str, data_freq: int = DATA_FREQ) -> None:
    """Plot measured tau against the fitted J*alpha + b*omega + c*sign(omega)."""
    omega = df[VELOCITY_COL].to_numpy(dtype=np.float64)
    alpha = df[ACCELERATION_COL].to_numpy(dtype=np.float64)
    tau = df[TAU_COL].to_numpy(dtype=np.float64)
    if params["sign_flipped"]:
        tau = -tau
    fitted = params["J"] * alpha + params["b"] * omega + params["c"] * np.sign(omega)
    time = np.arange(len(df)) / data_freq

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(time, tau, label="Measured", alpha=0.7)
    ax.plot(time, fitted, label="Fitted")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque [Nm]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    mcap_file_paths = [
        "/workspace/data/training_data/2026_10_07/rosbag2_2026_10_07-12_21_08_0.mcap",
    ]
    val_mcap_file_paths = []

    if not mcap_file_paths:
        print("No MCAP files provided; add current = 0 back-drive recordings to mcap_file_paths in train_inertia.py.")
        return

    print("Loading and processing MCAP files...")
    train_dataframes = load_mcap_dataframes_parallel_cached(mcap_file_paths, freq=DATA_FREQ)
    val_dataframes = load_mcap_dataframes_parallel_cached(val_mcap_file_paths, freq=DATA_FREQ) if val_mcap_file_paths else []
    for df in train_dataframes + val_dataframes:
        process_dataframe(df)

    train_samples = build_regression_samples(train_dataframes, VELOCITY_THRESHOLD)
    print(f"Using {len(train_samples['tau'])} training samples with |omega| >= {VELOCITY_THRESHOLD} rad/s")
    if len(train_samples["tau"]) == 0:
        print("No samples above the velocity threshold; cannot fit.")
        return

    fit = fit_inertia(train_samples["alpha"], train_samples["omega"], train_samples["tau"])
    print("\nGlobal fit (tau = J*alpha + b*omega + c*sign(omega)):")
    print(f"  J = {fit['J']:.6g} Nm*s^2/rad")
    print(f"  b = {fit['b']:.6g} Nm*s/rad")
    print(f"  c = {fit['c']:.6g} Nm")
    if fit["sign_flipped"]:
        print("  sensor torque sign flipped (fitted J was negative)")
    print(f"  RMSE: {fit['rmse']:.6g} Nm, R^2: {fit['r2']:.4f}")
    print(f"  regressor condition number: {fit['condition_number']:.3g}")
    print(f"  corr(alpha, omega): {fit['alpha_omega_correlation']:.3f}")
    print(f"  J*alpha share of mean |tau|: {fit['inertia_torque_share']:.3f}")

    per_file = {}
    print("\nPer-file fits (J consistency check):")
    for path, df in zip(mcap_file_paths, train_dataframes):
        name = os.path.basename(path)
        samples = build_regression_samples([df], VELOCITY_THRESHOLD)
        if len(samples["tau"]) == 0:
            print(f"  {name}: no samples above the velocity threshold")
            continue
        file_fit = fit_inertia(samples["alpha"], samples["omega"], samples["tau"])
        per_file[name] = {"J": file_fit["J"], "b": file_fit["b"], "c": file_fit["c"], "num_samples": int(len(samples["tau"]))}
        print(f"  {name}: J = {file_fit['J']:.6g}, b = {file_fit['b']:.6g}, c = {file_fit['c']:.6g} ({len(samples['tau'])} samples)")

    val_results = {}
    if val_dataframes:
        val_samples = build_regression_samples(val_dataframes, VELOCITY_THRESHOLD)
        if len(val_samples["tau"]) > 0:
            val_results = evaluate(fit, val_samples)
            val_results["num_samples"] = int(len(val_samples["tau"]))
            print(f"\nValidation: RMSE {val_results['rmse']:.6g} Nm, R^2 {val_results['r2']:.4f} ({val_results['num_samples']} samples)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "J": fit["J"],
        "b": fit["b"],
        "c": fit["c"],
        "sign_flipped": fit["sign_flipped"],
        "velocity_threshold_rad_per_sec": VELOCITY_THRESHOLD,
        "data_freq_hz": DATA_FREQ,
        "train": {
            "num_samples": int(len(train_samples["tau"])),
            "rmse": fit["rmse"],
            "r2": fit["r2"],
            "condition_number": fit["condition_number"],
            "alpha_omega_correlation": fit["alpha_omega_correlation"],
            "inertia_torque_share": fit["inertia_torque_share"],
            "per_file": per_file,
        },
        "val": val_results,
    }
    params_path = os.path.join(OUTPUT_DIR, "inertia_params.json")
    with open(params_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {params_path}")

    plot_df = val_dataframes[0] if val_dataframes else train_dataframes[0]
    plot_path = os.path.join(OUTPUT_DIR, "inertia_tau_fit.png")
    plot_tau_fit(plot_df, fit, plot_path, DATA_FREQ)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
