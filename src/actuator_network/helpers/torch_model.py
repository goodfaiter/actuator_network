import math

import torch

from actuator_network.helpers.m5_model import M5FrictionModel


class TorchMlpModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: list, device: torch.device):
        super(TorchMlpModel, self).__init__()
        layers = []
        in_size = input_size

        for hidden_size in hidden_layers:
            layers.append(torch.nn.Linear(in_size, hidden_size, device=device))
            layers.append(torch.nn.Tanh())
            in_size = hidden_size

        layers.append(torch.nn.Linear(in_size, output_size, device=device))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchRNNModel(torch.nn.Module):
    """GRU-based RNN model with PyTorch"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        device: torch.device,
        dropout: float = 0.1,
    ):
        super(TorchRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, device=device)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=output_size, device=device)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        return out, hn


class TorchTransformerModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        history_size: int,
        num_heads: int,
        hidden_dim: int,
        device: torch.device,
    ):
        super(TorchTransformerModel, self).__init__()

        # Input projection
        self.input_projection = torch.nn.Linear(input_size, hidden_dim, device=device)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(max_len=history_size, hidden_dim=hidden_dim, device=device)

        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            device=device,
            dropout=0.1,
            activation=torch.nn.ReLU(),
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output layer (taking only the last timestep)
        self.output_sequence = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 4, output_size, device=device),
        )

        # Causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(history_size).to(device)
        self.register_buffer("causal_mask", mask)

        # Store config
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, History, Feature Dim]
        #
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer processing
        x = self.transformer(x, mask=self.causal_mask, is_causal=True)
        # x = self.transformer(x)

        # Take only the last timestep and output
        x = x[:, -1, :]  # Take last timestep
        output = self.output_sequence(x)

        return output.unsqueeze(1)  # Unsqueeze to keep consistent output shape


class M5TransformerPhysicsModel(torch.nn.Module):
    """Transformer predicts tau_external, then M5 computes friction and physics yields the final force.

    The wrapped Transformer outputs a normalized tendon-force estimate. That estimate is
    denormalized and fed into the M5 friction model as tau_external. M5 returns tau_friction,
    and the final output is tau_external_calculated = tau_motor - tau_friction.

    The forward pass returns a 4-channel output:
        0: tau_external_calculated
        1: tau_motor
        2: tau_friction
        3: tau_external_pred
    """

    def __init__(
        self,
        m5: M5FrictionModel,
        transformer: TorchTransformerModel,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
        delta_position_idx: int,
        velocity_idx: int,
    ) -> None:
        super().__init__()
        self.m5 = m5
        self.transformer = transformer
        # normalize_tensor returns [1, feature_dim] statistics; flatten to 1D for indexing.
        self.register_buffer("input_mean", input_mean.view(-1))
        self.register_buffer("input_std", input_std.view(-1))
        self.register_buffer("output_mean", output_mean.view(-1))
        self.register_buffer("output_std", output_std.view(-1))
        self.delta_position_idx = delta_position_idx
        self.velocity_idx = velocity_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, History, Feature Dim] (normalized by ScaledModelWrapper)
        x_last = x[:, -1, :]

        # Un-normalize the last timestep for the physical M5 inputs
        delta_position_raw = (
            x_last[:, self.delta_position_idx] * self.input_std[self.delta_position_idx]
            + self.input_mean[self.delta_position_idx]
        )
        velocity_raw = (
            x_last[:, self.velocity_idx] * self.input_std[self.velocity_idx] + self.input_mean[self.velocity_idx]
        )
        tau_motor = self.m5.compute_tau_motor(delta_position_raw)

        # Transformer predicts the normalized tau_external
        tau_external_pred_norm = self.transformer(x)  # [Batch, 1, Output Dim]

        # Denormalize for M5, which expects physical units. The model is designed for a single
        # output column, so we index the first (and only) output statistic.
        tau_external_pred_phys = tau_external_pred_norm * self.output_std[0] + self.output_mean[0]
        tau_external_pred_phys = tau_external_pred_phys.squeeze(1).squeeze(1)

        # M5 predicts friction from velocity, motor torque, and predicted external torque
        tau_friction = self.m5(velocity_raw, tau_motor, tau_external_pred_phys)

        # Physics: tau_external = tau_motor - tau_friction
        tau_external_calc_phys = tau_motor - tau_friction

        # Normalize all quantities back so ScaledModelWrapper can denormalize consistently.
        # All four channels share the same physical unit, so they use the same mean/std.
        tau_external_calc_norm = (tau_external_calc_phys - self.output_mean[0]) / self.output_std[0]
        tau_motor_norm = (tau_motor - self.output_mean[0]) / self.output_std[0]
        tau_friction_norm = (tau_friction - self.output_mean[0]) / self.output_std[0]
        tau_external_pred_norm = (tau_external_pred_phys - self.output_mean[0]) / self.output_std[0]

        output = torch.stack(
            [tau_external_calc_norm, tau_motor_norm, tau_friction_norm, tau_external_pred_norm], dim=-1
        )
        return output.unsqueeze(1)  # [Batch, 1, 4]


class PlainM5PhysicsModel(torch.nn.Module):
    """M5 friction model as a standalone force estimator.

    Given delta_position and velocity, this model computes tau_motor and then
    solves ``tau_external = tau_motor - tau_friction(velocity, tau_motor, tau_external)``
    with a few fixed-point iterations. The forward pass returns a 4-channel output:

        0: tau_external_calculated
        1: tau_motor
        2: tau_friction
        3: tau_external_pred (identical to channel 0 for this model)
    """

    def __init__(self, m5: M5FrictionModel, num_iterations: int = 5) -> None:
        super().__init__()
        self.m5 = m5
        self.num_iterations = num_iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, 2] -> [delta_position, velocity]
        delta_position = x[:, 0]
        velocity = x[:, 1]

        tau_motor = self.m5.compute_tau_motor(delta_position)
        tau_external = tau_motor
        tau_friction = torch.zeros_like(tau_motor)
        for _ in range(self.num_iterations):
            tau_friction = self.m5(velocity, tau_motor, tau_external)
            tau_external = tau_motor - tau_friction

        output = torch.stack([tau_external, tau_motor, tau_friction, tau_external], dim=-1)
        return output.unsqueeze(1)  # [Batch, 1, 4]


class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 5000, device: torch.device = None):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, device=device) * (-math.log(10000.0) / hidden_dim))

        pe = torch.zeros(max_len, hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: [1, max_len, hidden_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, History, Hidden Dim]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
