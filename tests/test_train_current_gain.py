"""Tests for the current gain fitting pipeline."""

from unittest import mock

import numpy as np
import pandas as pd

from actuator_network.train_current_gain import (
    CURRENT_COL,
    DEFAULT_FRICTION_PARAMS,
    DESIRED_VELOCITY_COL,
    POSITION_COL,
    TAU_COL,
    VELOCITY_COL,
    build_gain_points,
    fit_gain,
    load_friction_params,
    main,
    per_level_gains,
    segment_plateaus,
)


def _make_processed_df(n: int = 300, gain: float = 2.0, tau_f: float = 0.01, tau_sign: float = -1.0) -> pd.DataFrame:
    """Synthetic processed DataFrame for one wind/unwind plateau pair."""
    position = np.concatenate([np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n)])
    tau_ext = 0.1 * position
    desired = np.concatenate([np.full(n, 0.1), np.full(n, -0.1)])
    measured = desired.copy()
    current = np.concatenate([(tau_ext[:n] + tau_f) / gain, (tau_ext[n:] - tau_f) / gain])
    tau = tau_sign * tau_ext
    return pd.DataFrame(
        {
            CURRENT_COL: current,
            DESIRED_VELOCITY_COL: desired,
            VELOCITY_COL: measured,
            POSITION_COL: position,
            TAU_COL: tau,
            "desired_position_rad_data": np.zeros(2 * n),
            "weight_kg_data": np.zeros(2 * n),
        },
        index=pd.to_timedelta(np.arange(2 * n) * 5.0, unit="ms"),
    )


def test_segment_plateaus_excludes_static_short_runs_and_ramps():
    desired = np.concatenate([np.zeros(10), np.full(200, 0.1), [0.05], np.full(200, 0.2), np.full(20, 0.3), np.zeros(10)])
    measured = desired.copy()

    plateaus = segment_plateaus(desired, measured, min_samples=100, velocity_threshold=0.05)

    assert [level for level, _ in plateaus] == [0.1, 0.2]
    for _, mask in plateaus:
        assert mask.sum() == 200
        assert (np.abs(measured[mask]) >= 0.05).all()


def test_build_gain_points_and_fit_recovers_gain():
    gain, tau_f = 2.0, 0.01
    df = _make_processed_df(gain=gain, tau_f=tau_f)

    points = build_gain_points([df], velocity_threshold=0.05, min_samples=100, num_bins=10)
    fit = fit_gain(points)

    assert not fit["sign_flipped"]
    np.testing.assert_allclose(fit["K_t"], gain, rtol=1e-6)
    assert fit["r2"] > 0.99
    # Friction cancels in the mean current and reappears in the half difference.
    np.testing.assert_allclose(gain * points["i_half_diff"].mean(), tau_f, rtol=1e-6)


def test_fit_gain_flips_flipped_load_sign():
    df = _make_processed_df(gain=2.0, tau_f=0.01, tau_sign=+1.0)

    points = build_gain_points([df], velocity_threshold=0.05, min_samples=100, num_bins=10)
    fit = fit_gain(points)

    assert fit["sign_flipped"]
    np.testing.assert_allclose(fit["K_t"], 2.0, rtol=1e-6)


def test_per_level_gains_capture_velocity_dependence():
    n = 300
    position = np.concatenate([np.linspace(0.0, 1.0, n)] * 4)
    tau_ext = 0.1 * position
    desired = np.concatenate([np.full(n, 0.1), np.full(n, -0.1), np.full(n, 0.2), np.full(n, -0.2)])
    measured = desired.copy()
    gains = {0.1: 1.0, 0.2: 2.0}
    frictions = {0.1: 0.01, 0.2: 0.02}
    current = []
    for level in (0.1, -0.1, 0.2, -0.2):
        sign = 1.0 if level > 0 else -1.0
        current.append((tau_ext[:n] + sign * frictions[abs(level)]) / gains[abs(level)])
    current = np.concatenate(current)
    df = pd.DataFrame(
        {
            CURRENT_COL: current,
            DESIRED_VELOCITY_COL: desired,
            VELOCITY_COL: measured,
            POSITION_COL: position,
            TAU_COL: -tau_ext,
        }
    )

    points = build_gain_points([df], velocity_threshold=0.05, min_samples=100, num_bins=10)
    fit = fit_gain(points)
    level_gains = per_level_gains(points, fit, {"b": 0.0, "c": 0.0})

    assert 1.0 < fit["K_t"] < 2.0
    np.testing.assert_allclose(level_gains[0.1]["K_t"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(level_gains[0.2]["K_t"], 2.0, rtol=1e-6)
    np.testing.assert_allclose(level_gains[0.1]["tau_f"], 0.01, rtol=1e-6)
    np.testing.assert_allclose(level_gains[0.2]["tau_f"], 0.02, rtol=1e-6)


def test_load_friction_params_falls_back_to_defaults(tmp_path):
    params = load_friction_params(params_path=str(tmp_path / "missing.json"))

    assert params == DEFAULT_FRICTION_PARAMS


def test_main_end_to_end(tmp_path):
    df = _make_processed_df()

    with (
        mock.patch("actuator_network.train_current_gain.load_mcap_dataframes_parallel_cached", return_value=[df]),
        mock.patch("actuator_network.train_current_gain.OUTPUT_DIR", str(tmp_path)),
    ):
        main()

    assert (tmp_path / "current_gain_params.json").is_file()
    assert (tmp_path / "current_gain_fit.png").is_file()
    assert (tmp_path / "current_gain_timeseries.png").is_file()
