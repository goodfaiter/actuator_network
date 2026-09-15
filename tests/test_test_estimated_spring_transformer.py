"""Tests for the estimated-spring transformer inference script."""

import torch

from actuator_network.test_estimated_spring_transformer import _build_inference_window


def test_build_inference_window_zero_pads_early_samples():
    """Early timesteps should receive zero-padded history windows."""
    features = torch.arange(1, 21, dtype=torch.float32).view(10, 2)
    num_hist = 3
    stride = 2
    device = torch.device("cpu")

    window = _build_inference_window(features, t=0, num_hist=num_hist, stride=stride, device=device)
    assert window.shape == (1, num_hist, 2)
    # The current timestep is at the end; earlier entries are zero-padded.
    assert torch.allclose(window[0, -1], features[0])
    assert torch.allclose(window[0, :-1], torch.zeros_like(window[0, :-1]))

    # The last entry of the window at timestep t should always be features[t].
    for t in range(features.shape[0]):
        window = _build_inference_window(features, t=t, num_hist=num_hist, stride=stride, device=device)
        assert torch.allclose(window[0, -1], features[t])

    # Once the window is fully inside the data, no zero padding remains.
    fully_inside_t = (num_hist - 1) * stride
    window = _build_inference_window(features, t=fully_inside_t, num_hist=num_hist, stride=stride, device=device)
    assert torch.all(window != 0.0)
