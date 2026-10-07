"""Tests for the inertia fitting pipeline."""

import numpy as np
import pandas as pd
import pytest

from actuator_network.train_inertia import (
    ACCELERATION_COL,
    TAU_COL,
    VELOCITY_COL,
    build_regression_samples,
    evaluate,
    fit_inertia,
    main,
)


def _make_df(velocity, acceleration, tau) -> pd.DataFrame:
    return pd.DataFrame({VELOCITY_COL: velocity, ACCELERATION_COL: acceleration, TAU_COL: tau})


def _synthetic_samples(rng: np.random.Generator, n: int, inertia: float, b: float, c: float, noise_std: float = 0.0):
    omega = rng.uniform(-3.0, 3.0, n)
    alpha = rng.uniform(-40.0, 40.0, n)
    tau = inertia * alpha + b * omega + c * np.sign(omega)
    if noise_std > 0:
        tau = tau + rng.normal(0.0, noise_std, n)
    return alpha, omega, tau


def test_fit_inertia_recovers_exact_parameters():
    rng = np.random.default_rng(0)
    inertia, b, c = 0.002, 0.002, 0.02
    alpha, omega, tau = _synthetic_samples(rng, 5000, inertia, b, c)

    fit = fit_inertia(alpha, omega, tau)

    assert not fit["sign_flipped"]
    np.testing.assert_allclose(fit["J"], inertia, rtol=1e-8)
    np.testing.assert_allclose(fit["b"], b, rtol=1e-8)
    np.testing.assert_allclose(fit["c"], c, rtol=1e-8)
    assert fit["rmse"] < 1e-12
    assert fit["r2"] > 0.999999


def test_fit_inertia_recovers_noisy_parameters():
    rng = np.random.default_rng(1)
    inertia, b, c = 0.002, 0.002, 0.02
    alpha, omega, tau = _synthetic_samples(rng, 20000, inertia, b, c, noise_std=0.002)

    fit = fit_inertia(alpha, omega, tau)

    assert not fit["sign_flipped"]
    assert abs(fit["J"] - inertia) < 1e-5
    assert abs(fit["b"] - b) < 1e-4
    assert abs(fit["c"] - c) < 1e-4
    assert fit["r2"] > 0.99


def test_fit_inertia_flips_flipped_sensor_sign():
    rng = np.random.default_rng(2)
    inertia, b, c = 0.002, 0.002, 0.02
    alpha, omega, tau = _synthetic_samples(rng, 5000, inertia, b, c)

    fit = fit_inertia(alpha, omega, -tau)

    assert fit["sign_flipped"]
    np.testing.assert_allclose(fit["J"], inertia, rtol=1e-8)
    np.testing.assert_allclose(fit["b"], b, rtol=1e-8)
    np.testing.assert_allclose(fit["c"], c, rtol=1e-8)


def test_build_regression_samples_filters_static_and_nonfinite():
    # The first sample is a derivative fill artifact and is always dropped.
    velocity = [0.5, 0.04, 0.06, -0.06, 0.5]
    acceleration = [0.0, 1.0, 2.0, 3.0, 4.0]
    tau = [1.0, 2.0, 3.0, 4.0, np.nan]
    df = _make_df(velocity, acceleration, tau)

    samples = build_regression_samples([df], velocity_threshold=0.05)

    np.testing.assert_allclose(samples["omega"], [0.06, -0.06])
    np.testing.assert_allclose(samples["alpha"], [2.0, 3.0])
    np.testing.assert_allclose(samples["tau"], [3.0, 4.0])


def test_evaluate_applies_sensor_sign_flip():
    samples = {"alpha": np.array([1.0, -1.0]), "omega": np.array([1.0, -1.0]), "tau": np.array([-1.0, 1.0])}
    params = {"J": 0.0, "b": 0.0, "c": 1.0, "sign_flipped": True}

    metrics = evaluate(params, samples)

    assert metrics["rmse"] == pytest.approx(0.0)


def test_main_without_files_prints_hint(capsys):
    main()

    assert "No MCAP files provided" in capsys.readouterr().out
