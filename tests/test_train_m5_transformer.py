import torch

from actuator_network.helpers.m5_model import M5FrictionModel
from actuator_network.helpers.torch_model import M5TransformerPhysicsModel, TorchTransformerModel


def _make_model(input_dim: int = 2, history_size: int = 8, output_dim: int = 1):
    """Create a small M5TransformerPhysicsModel for testing."""
    m5 = M5FrictionModel()
    transformer = TorchTransformerModel(
        input_size=input_dim,
        output_size=output_dim,
        num_layers=1,
        history_size=history_size,
        num_heads=2,
        hidden_dim=8,
        device=torch.device("cpu"),
    )
    input_mean = torch.zeros(input_dim)
    input_std = torch.ones(input_dim)
    output_mean = torch.zeros(output_dim)
    output_std = torch.ones(output_dim)

    model = M5TransformerPhysicsModel(
        m5=m5,
        transformer=transformer,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        delta_position_idx=0,
        velocity_idx=1,
    )
    return model


def test_m5_transformer_physics_forward_shape():
    """The combined model should return [Batch, 1, Output Dim]."""
    model = _make_model()
    x = torch.randn(4, 8, 2)
    out = model(x)
    assert out.shape == (4, 1, 1)


def test_m5_transformer_physics_computes_tau_external():
    """With a deterministic M5 (constant friction), output equals tau_motor - tau_friction."""
    m5 = M5FrictionModel()
    # Make M5 return a constant friction of 0.5
    with torch.no_grad():
        m5.Kv.fill_(0.0)
        m5.Kc.fill_(0.5)
        m5.Km.fill_(0.0)
        m5.Ke.fill_(0.0)
        m5.Kcs.fill_(0.0)
        m5.K_ms.fill_(0.0)
        m5.Kes.fill_(0.0)
        # Make the Stribeck envelope essentially 1.0 for any velocity
        m5.V_s_log.fill_(20.0)
        m5.alpha_log.fill_(1.0)

    transformer = TorchTransformerModel(
        input_size=2,
        output_size=1,
        num_layers=1,
        history_size=8,
        num_heads=2,
        hidden_dim=8,
        device=torch.device("cpu"),
    )

    input_mean = torch.zeros(2)
    input_std = torch.ones(2)
    output_mean = torch.tensor([0.5])
    output_std = torch.tensor([2.0])

    model = M5TransformerPhysicsModel(
        m5=m5,
        transformer=transformer,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        delta_position_idx=0,
        velocity_idx=1,
        motor_gain=4.2,
    )

    batch = 4
    history = 8
    x = torch.randn(batch, history, 2)
    # delta_position is the first feature at the last timestep
    delta_position = x[:, -1, 0]
    expected_phys = 4.2 * delta_position - 0.5
    expected_norm = (expected_phys - output_mean) / output_std

    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch, 1, 1)
    assert torch.allclose(out.squeeze(), expected_norm, atol=1e-6)


def test_m5_transformer_physics_gradients_flow_to_transformer():
    """Backpropagation should reach the Transformer parameters through M5."""
    model = _make_model()
    model.m5.requires_grad_(False)

    x = torch.randn(4, 8, 2)
    target = torch.randn(4, 1, 1)

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    transformer_params = list(model.transformer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in transformer_params)
