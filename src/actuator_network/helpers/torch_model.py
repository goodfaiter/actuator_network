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
        dropout: float = 0.1,
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
            dropout=dropout,
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
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
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


class SpringTransformerForceEstimator(torch.nn.Module):
    """Stateful deployable model: spring-class estimator + force estimator.

    The model receives a sliding window of ``[delta_position, velocity]`` sampled
    at ``force_stride``. It maintains its own internal spring input buffer sampled
    at ``spring_stride``. When the last-step ``|velocity|`` exceeds
    ``velocity_threshold`` and the current call coincides with a spring sample
    instant, the buffer is shifted and the current sample appended; otherwise the
    buffer is held frozen. The spring transformer is run on this buffer to produce
    a spring estimate, which is repeated across the history dimension and
    concatenated to the incoming ``[delta_position, velocity]`` window before the
    force transformer predicts ``tendon_bota_force_newton_data``.

    All internal transformer inputs/outputs are normalized. The wrapper handles
    input normalization and output denormalization.

    Important: this model is designed for online inference with **batch size 1**.
    ``spring_stride`` must be a multiple of ``force_stride`` and at least as large.

    The forward pass returns a 2-channel output:
        0: ``tendon_bota_force_newton_data`` (normalized with force output stats)
        1: ``spring_coeff`` (normalized with spring output stats)
    """

    def __init__(
        self,
        spring_transformer: TorchTransformerModel,
        force_transformer: TorchTransformerModel,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        spring_input_mean: torch.Tensor,
        spring_input_std: torch.Tensor,
        velocity_idx: int = 1,
        velocity_threshold: float = 0.1,
        spring_alpha: float = 0.9,
        spring_stride: int = 1,
        force_stride: int = 1,
    ) -> None:
        super().__init__()
        self.spring_transformer = spring_transformer
        self.force_transformer = force_transformer

        # Each transformer may have its own history size and stride.
        spring_history_size = int(spring_transformer.causal_mask.size(0))
        force_history_size = int(force_transformer.causal_mask.size(0))
        hidden_dim = spring_transformer.hidden_dim

        if spring_stride % force_stride != 0:
            raise ValueError(f"spring_stride ({spring_stride}) must be a multiple of force_stride ({force_stride})")
        if spring_stride < force_stride:
            raise ValueError(f"spring_stride ({spring_stride}) must be >= force_stride ({force_stride})")

        self.register_buffer("input_mean", input_mean.view(-1))
        self.register_buffer("input_std", input_std.view(-1))
        self.register_buffer("spring_input_mean", spring_input_mean.view(-1))
        self.register_buffer("spring_input_std", spring_input_std.view(-1))
        self.register_buffer("velocity_threshold", torch.tensor(velocity_threshold, dtype=torch.float32))
        self.register_buffer("spring_alpha", torch.tensor(spring_alpha, dtype=torch.float32))
        self.register_buffer("spring_stride", torch.tensor(spring_stride, dtype=torch.int64))
        self.register_buffer("force_stride", torch.tensor(force_stride, dtype=torch.int64))
        self.register_buffer("spring_update_counter", torch.zeros(1, dtype=torch.int64))
        self.velocity_idx = velocity_idx
        self.spring_history_size = spring_history_size
        self.force_history_size = force_history_size
        self.hidden_dim = hidden_dim

        # Stateful buffers for online inference. The spring buffer stores
        # normalized values; initialize it to the normalized value of physical
        # zero so it matches the zero-filled training windows.
        spring_buffer_zero = (0.0 - self.spring_input_mean) / self.spring_input_std
        self.register_buffer(
            "spring_buffer", spring_buffer_zero.view(1, 1, -1).expand(1, spring_history_size, -1).clone()
        )
        self.register_buffer("last_spring", torch.zeros(1, 1, 1))

    def reset(self) -> None:
        """Clear the internal spring buffer and last spring estimate."""
        spring_buffer_zero = (0.0 - self.spring_input_mean) / self.spring_input_std
        self.spring_buffer.copy_(spring_buffer_zero.view(1, 1, -1).expand(1, self.spring_history_size, -1))
        self.last_spring.zero_()
        self.spring_update_counter.zero_()

    def _is_moving(self, velocity: torch.Tensor) -> torch.Tensor:
        return torch.abs(velocity) > self.velocity_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [1, History, Feature] (already normalized by ScaledModelWrapper)
        # Recover physical velocity only for the moving threshold check.
        last_velocity = (
            x[0, -1, self.velocity_idx] * self.input_std[self.velocity_idx] + self.input_mean[self.velocity_idx]
        )

        spring_update_ratio = int(self.spring_stride.item() // self.force_stride.item())
        is_spring_sample = int(self.spring_update_counter.item()) % spring_update_ratio == 0

        if is_spring_sample and self._is_moving(last_velocity):
            # Shift the spring buffer and append the current normalized sample.
            self.spring_buffer[0, :-1, :] = self.spring_buffer[0, 1:, :].clone()
            self.spring_buffer[0, -1, :] = x[0, -1, :]
        # else: keep the previous spring buffer frozen.

        self.spring_update_counter.add_(1)

        # Run spring transformer on the normalized spring buffer.
        spring_pred_norm = self.spring_transformer(self.spring_buffer)  # [1, 1, 1]

        # Smooth the spring estimate with exponential moving average to discourage
        # rapid switching between spring predictions.
        smoothed_spring = self.spring_alpha * spring_pred_norm + (1.0 - self.spring_alpha) * self.last_spring
        self.last_spring.copy_(smoothed_spring)

        # Build force transformer input: [delta_position, velocity, spring].
        # The incoming force window may be shorter than the spring buffer, so
        # expand the spring estimate to match the force transformer's input length.
        spring_channel = smoothed_spring.expand(1, x.size(1), -1)
        force_input_norm = torch.cat([x, spring_channel], dim=-1)  # [1, History, 3]
        force_pred_norm = self.force_transformer(force_input_norm)  # [1, 1, 1]

        # Stack force and spring predictions so the wrapper can denormalize each
        # channel with its own output statistics.
        return torch.cat([force_pred_norm, smoothed_spring], dim=-1)  # [1, 1, 2]


class SpringForceTrainingModel(torch.nn.Module):
    """Joint training wrapper for the spring and force transformers.

    All inputs and outputs are expected to be already normalized. The input is
    a pair of tensors ``(spring_windows, force_windows)``. The spring windows
    may be longer than the force windows; the spring prediction is repeated to
    match the force window length. The output is a single tensor of shape
    ``[Batch, 1, 2]``:
        0: predicted force (normalized with force output stats)
        1: predicted spring coefficient (normalized with spring output stats)
    """

    def __init__(
        self,
        spring_transformer: TorchTransformerModel,
        force_transformer: TorchTransformerModel,
    ) -> None:
        super().__init__()
        self.spring_transformer = spring_transformer
        self.force_transformer = force_transformer

        self.spring_history_size = int(spring_transformer.causal_mask.size(0))
        self.force_history_size = int(force_transformer.causal_mask.size(0))

    def forward(self, spring_windows: torch.Tensor, force_windows: torch.Tensor) -> torch.Tensor:
        # spring_windows shape: [Batch, Spring History, Feature Dim]
        # force_windows shape: [Batch, Force History, Feature Dim]

        # Spring prediction (input is already normalized).
        spring_pred_norm = self.spring_transformer(spring_windows)  # [Batch, 1, 1]

        # The spring estimate is already normalized in spring-output space.
        # Repeat it across the force history dimension and feed it to the force transformer.
        spring_channel = spring_pred_norm.expand(-1, force_windows.size(1), -1)
        force_input_norm = torch.cat([force_windows, spring_channel], dim=-1)
        force_pred_norm = self.force_transformer(force_input_norm)  # [Batch, 1, 1]

        return torch.cat([force_pred_norm, spring_pred_norm], dim=-1)  # [Batch, 1, 2]


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
