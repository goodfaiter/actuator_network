import torch

from actuator_network.helpers.m5_model import M5FrictionModel


def test_m5_forward_matches_formula():
    """The model output should match a manual implementation of the M5 formula."""
    model = M5FrictionModel()

    # Set known parameters
    with torch.no_grad():
        model.Kv.fill_(0.1)
        model.Kc.fill_(0.2)
        model.Km.fill_(0.3)
        model.Ke.fill_(0.4)
        model.V_s_log.fill_(0.0)  # softplus(0) = ln(2) ~= 0.693
        model.alpha_log.fill_(0.0)
        model.Kcs.fill_(0.5)
        model.K_ms.fill_(0.6)
        model.Kes.fill_(0.7)

    velocity = torch.tensor([0.0, 1.0, -2.0])
    tau_motor = torch.tensor([0.5, -0.5, 1.0])
    tau_external = torch.tensor([0.0, 1.0, -0.5])

    prediction = model(velocity, tau_motor, tau_external)

    # Manual computation
    v_s = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-6
    alpha = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-6
    static = (
        0.1 * velocity
        + 0.2
        + torch.abs(0.3 * tau_motor - 0.4 * tau_external)
    )
    stribeck = torch.exp(-torch.abs(velocity / v_s).clamp_min(1e-8) ** alpha)
    expected = static + stribeck * (
        0.5 + torch.abs(0.6 * tau_motor - 0.7 * tau_external)
    )

    assert torch.allclose(prediction, expected, atol=1e-6)


def test_m5_positive_parameters_are_enforced():
    """V_s and alpha returned by the model must be positive."""
    model = M5FrictionModel()

    # Set unconstrained log parameters to negative values
    with torch.no_grad():
        model.V_s_log.fill_(-5.0)
        model.alpha_log.fill_(-10.0)

    params = model.named_physical_parameters()

    assert params["V_s"] > 0.0
    assert params["alpha"] > 0.0


def test_m5_fitting_reduces_loss():
    """A few Adam steps should reduce MSE on synthetic data."""
    torch.manual_seed(0)
    model = M5FrictionModel()

    velocity = torch.randn(100)
    tau_motor = torch.randn(100)
    tau_external = torch.randn(100)

    # Synthetic target from a known parameter set
    with torch.no_grad():
        target_model = M5FrictionModel()
        target_model.Kv.fill_(0.1)
        target_model.Kc.fill_(0.2)
        target_model.Km.fill_(0.3)
        target_model.Ke.fill_(0.4)
        target = target_model(velocity, tau_motor, tau_external)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial_loss = torch.nn.functional.mse_loss(model(velocity, tau_motor, tau_external), target).item()

    for _ in range(200):
        optimizer.zero_grad()
        pred = model(velocity, tau_motor, tau_external)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(model(velocity, tau_motor, tau_external), target).item()
    assert final_loss < initial_loss
