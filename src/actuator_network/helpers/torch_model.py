import math

import torch

from actuator_network.helpers.m5_model import M5FrictionModel


def _get_activation(activation: str) -> torch.nn.Module:
    """Return a PyTorch activation module from its name."""
    activations = {
        "relu": torch.nn.ReLU(),
        "tanh": torch.nn.Tanh(),
        "gelu": torch.nn.GELU(),
        "leaky_relu": torch.nn.LeakyReLU(),
        "elu": torch.nn.ELU(),
        "silu": torch.nn.SiLU(),
    }
    activation = activation.lower()
    if activation not in activations:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activations.keys())}")
    return activations[activation]


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
        activation: str = "relu",
    ):
        super(TorchTransformerModel, self).__init__()

        activation_module = _get_activation(activation)

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
            activation=activation_module,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output layer (taking only the last timestep)
        self.output_sequence = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            _get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            _get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 4, output_size, device=device),
        )

        # Causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(history_size).to(device)
        self.register_buffer("causal_mask", mask)

        # Store config
        self.input_size = input_size
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

        # Take only the last timestep and output
        x = x[:, -1, :]  # Take last timestep
        output = self.output_sequence(x)

        return output.unsqueeze(1)  # Unsqueeze to keep consistent output shape


class M5TransformerPhysicsModel(torch.nn.Module):
    """Transformer predicts tau_external, then M5 computes friction and physics yields the final force.

    The wrapped Transformer outputs a normalized tendon-force estimate. That estimate is
    denormalized and fed into the M5 friction model as tau_external. M5 returns tau_friction,
    and the final output is tau_external_calculated = tau_motor - tau_friction.

    Normalization statistics are not stored by this model; the ScaledModelWrapper passes
    its own buffers (flattened to 1-D) into the forward call, so the wrapper is the single
    source of truth for normalization.

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
        delta_position_idx: int,
        velocity_idx: int,
    ) -> None:
        super().__init__()
        self.m5 = m5
        self.transformer = transformer
        self.delta_position_idx = delta_position_idx
        self.velocity_idx = velocity_idx

    def forward(
        self,
        x: torch.Tensor,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> torch.Tensor:
        # x shape: [Batch, History, Feature Dim] (normalized by ScaledModelWrapper)
        # input_mean/std: [Feature Dim]; output_mean/std: [Output Channels]
        x_last = x[:, -1, :]

        # Un-normalize the last timestep for the physical M5 inputs
        delta_position_raw = (
            x_last[:, self.delta_position_idx] * input_std[self.delta_position_idx]
            + input_mean[self.delta_position_idx]
        )
        velocity_raw = x_last[:, self.velocity_idx] * input_std[self.velocity_idx] + input_mean[self.velocity_idx]
        tau_motor = self.m5.compute_tau_motor(delta_position_raw)

        # Transformer predicts the normalized tau_external
        tau_external_pred_norm = self.transformer(x)  # [Batch, 1, Output Dim]

        # Denormalize for M5, which expects physical units. The model is designed for a single
        # output column, so we index the first (and only) output statistic.
        tau_external_pred_phys = tau_external_pred_norm * output_std[0] + output_mean[0]
        tau_external_pred_phys = tau_external_pred_phys.squeeze(1).squeeze(1)

        # M5 predicts friction from velocity, motor torque, and predicted external torque
        tau_friction = self.m5(velocity_raw, tau_motor, tau_external_pred_phys)

        # Physics: tau_external = tau_motor - tau_friction
        tau_external_calc_phys = tau_motor - tau_friction

        # Normalize all quantities back so ScaledModelWrapper can denormalize consistently.
        # All four channels share the same physical unit, so they use the same mean/std.
        tau_external_calc_norm = (tau_external_calc_phys - output_mean[0]) / output_std[0]
        tau_motor_norm = (tau_motor - output_mean[0]) / output_std[0]
        tau_friction_norm = (tau_friction - output_mean[0]) / output_std[0]
        tau_external_pred_norm = (tau_external_pred_phys - output_mean[0]) / output_std[0]

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


class SpringCoefficientHead(torch.nn.Module):
    """Small MLP that reconstructs a spring coefficient from a latent vector.

    Architecture: ``latent_dim -> 4 -> 1`` with ReLU activation on the hidden
    layer. This head provides the auxiliary spring-coefficient target while the
    main latent representation is fed to the force transformer.
    """

    def __init__(self, latent_dim: int, device: torch.device):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 4, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, 1, latent_dim]
        return self.network(x)


class SpringTransformerModel(torch.nn.Module):
    """Spring + force transformer pair for training and stateful online deployment.

    The class serves both regimes with one dispatched forward:
    batched teacher-forced training (``forward(spring_windows, force_windows)``)
    and stateful online inference (``forward(x)`` with ``num_envs`` parallel
    environments). The underlying transformers are shared, so training through
    the batched path updates the weights used by the online path.

    The online path receives the normalized current sample(s) of shape
    ``[num_envs, 1, Feature]`` (``[delta_position, velocity, ...]``), already
    normalized by ``ScaledModelWrapper``. It maintains per-environment internal
    force and spring input buffers (both zero-initialized to match the
    zero-padded training windows built in the normalized domain). One call is
    one force tick: the force buffer appends the current sample, and the spring
    buffer is shifted and appended only when the environment is moving
    (``|velocity|`` outside the prescaled threshold bounds ``velocity_threshold_hi``/
    ``velocity_threshold_lo``, derived at construction time from the physical
    threshold and the training statistics) and the current call coincides with a
    spring sample instant; otherwise the spring buffer is held frozen. The model
    transformer runs on the spring buffer to produce a latent representation,
    which is EMA-smoothed, repeated across the force window, and concatenated to
    the internal force buffer before the force transformer predicts
    ``tendon_bota_force_newton_data``. A small ``SpringCoefficientHead``
    reconstructs the spring coefficient from the latent vector for the auxiliary
    output channel. ``reset(reset_idx)`` clears the per-environment states.

    The spring transformer is called only on ticks where at least one
    environment updates its spring buffer or has not produced a latent yet
    (fresh/reset state); on all-frozen ticks the latent EMA continues from the
    cached latent of the last transformer run, which is numerically identical
    to re-running the transformer on the unchanged buffer. The buffer update
    is branch-free (``torch.where`` instead of boolean-mask indexing) and the
    per-tick spring sample ratio and EMA weight are stored as host-side
    constants, so the scripted forward never inserts a device synchronization
    to read them.

    The wrapper handles input normalization and output denormalization.

    Important: this model is designed for online inference with arbitrary batch
    size ``num_envs``. Each call is one force tick: the internal force buffer
    appends the current sample, and the spring counter ticks (a spring buffer
    update happens every ``spring_stride // force_stride`` calls when the
    environment is moving). ``spring_stride`` must be a multiple of
    ``force_stride`` and at least as large.

    The forward pass returns a 2-channel output:
        0: ``tendon_bota_force_newton_data`` (normalized with force output stats)
        1: ``spring_coeff`` (normalized with spring output stats)
    """

    def __init__(
        self,
        model_transformer: TorchTransformerModel,
        force_transformer: TorchTransformerModel,
        spring_coeff_head: SpringCoefficientHead,
        latent_dim: int,
        velocity_idx: int = 1,
        velocity_threshold_lo: float = -0.1,
        velocity_threshold_hi: float = 0.1,
        spring_alpha: float = 0.9,
        spring_stride: int = 1,
        force_stride: int = 1,
    ) -> None:
        super().__init__()
        self.model_transformer = model_transformer
        self.force_transformer = force_transformer
        self.spring_coeff_head = spring_coeff_head
        self.latent_dim = latent_dim

        # Each transformer may have its own history size and stride.
        spring_history_size = int(model_transformer.causal_mask.size(0))
        force_history_size = int(force_transformer.causal_mask.size(0))
        hidden_dim = model_transformer.hidden_dim

        if spring_stride % force_stride != 0:
            raise ValueError(f"spring_stride ({spring_stride}) must be a multiple of force_stride ({force_stride})")
        if spring_stride < force_stride:
            raise ValueError(f"spring_stride ({spring_stride}) must be >= force_stride ({force_stride})")

        # Host-side constants: kept out of the per-tick loop so the scripted
        # forward never inserts a device synchronization to read them.
        self.spring_update_ratio: int = spring_stride // force_stride
        self.spring_alpha: float = spring_alpha

        self.register_buffer("velocity_threshold_lo", torch.tensor(velocity_threshold_lo, dtype=torch.float32))
        self.register_buffer("velocity_threshold_hi", torch.tensor(velocity_threshold_hi, dtype=torch.float32))
        self.register_buffer("spring_stride", torch.tensor(spring_stride, dtype=torch.int64))
        self.register_buffer("force_stride", torch.tensor(force_stride, dtype=torch.int64))
        self.velocity_idx = velocity_idx
        self.spring_history_size = spring_history_size
        self.force_history_size = force_history_size
        self.hidden_dim = hidden_dim

        # Stateful buffers for online inference, one state per environment.
        # The buffers are registered with a single environment and the compiled
        # model reassigns them to the incoming batch size, so the scripted
        # model works with any num_envs. They store normalized values and are
        # zero-initialized to match the zero-padded training windows built in
        # the normalized domain.
        input_size = model_transformer.input_size
        self.register_buffer("spring_buffer", torch.zeros(1, spring_history_size, input_size))
        self.register_buffer("force_buffer", torch.zeros(1, force_history_size, input_size))
        self.register_buffer("last_latent", torch.zeros(1, 1, latent_dim))
        self.register_buffer("spring_update_counter", torch.zeros(1, dtype=torch.int64))
        # Cached transformer output per env (used by the all-frozen fast path)
        # and validity flag: an env that never ran the transformer (fresh or
        # reset state) must run it once before the fast path may serve it.
        self.register_buffer("latent_anchor", torch.zeros(1, 1, latent_dim))
        self.register_buffer("anchor_valid", torch.zeros(1, dtype=torch.bool))

    def _ensure_state(self, num_envs: int, device: torch.device) -> None:
        """Re-initialize the per-env state to zeros when the input batch size changes.

        Changing the batch size resets all environment states (fresh session
        semantics); the zero initialization matches the zero-padded training
        windows built in the normalized domain.
        """
        input_size = self.model_transformer.input_size
        if self.spring_buffer.size(0) != num_envs:
            self.spring_buffer = torch.zeros(num_envs, self.spring_history_size, input_size, device=device)
            self.force_buffer = torch.zeros(num_envs, self.force_history_size, input_size, device=device)
            self.last_latent = torch.zeros(num_envs, 1, self.latent_dim, device=device)
            self.spring_update_counter = torch.zeros(num_envs, dtype=torch.int64, device=device)
            self.latent_anchor = torch.zeros(num_envs, 1, self.latent_dim, device=device)
            self.anchor_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @torch.jit.export
    def reset(self, reset_idx: torch.Tensor | None = None) -> None:
        """Clear the internal buffers and counters.

        Args:
            reset_idx: Optional bool tensor of length ``num_envs`` selecting the
                environments to reset. When None, all environments are reset
                (mirroring ``TimeSeriesBuffer.reset_idx``).
        """
        if reset_idx is None:
            self.spring_buffer.zero_()
            self.force_buffer.zero_()
            self.last_latent.zero_()
            self.spring_update_counter.zero_()
            self.latent_anchor.zero_()
            self.anchor_valid.zero_()
            return
        mask = reset_idx.reshape(-1)
        self.spring_buffer[mask] = 0.0
        self.force_buffer[mask] = 0.0
        self.last_latent[mask] = 0.0
        self.spring_update_counter[mask] = 0
        self.latent_anchor[mask] = 0.0
        self.anchor_valid = self.anchor_valid & torch.logical_not(mask)

    def _is_moving(self, velocity: torch.Tensor) -> torch.Tensor:
        return (velocity > self.velocity_threshold_hi) | (velocity < self.velocity_threshold_lo)

    def forward(self, x: torch.Tensor, force_windows: torch.Tensor | None = None) -> torch.Tensor:
        """Run the model: stateful online inference, or batched teacher-forced training.

        With a single tensor, ``x`` is the normalized current sample(s) of shape
        ``[num_envs, 1, Feature]`` and the stateful online path is used (one
        internal force history buffer + spring state per environment). With two
        tensors, ``x`` is a batch of normalized spring windows of shape
        ``[Batch, Spring History, Feature]`` and ``force_windows`` the matching
        batch of force windows of shape ``[Batch, Force History, Feature]``;
        the batched teacher-forced path is used and no state is touched.
        """
        if force_windows is None:
            return self._forward_stateful(x)
        return self._forward_batched(x, force_windows)

    def _forward_stateful(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [num_envs, 1, Feature] (already normalized by ScaledModelWrapper)
        self._ensure_state(x.shape[0], x.device)
        last_velocity = x[:, -1, self.velocity_idx]

        is_spring_sample = (self.spring_update_counter % self.spring_update_ratio) == 0
        spring_mask = is_spring_sample & self._is_moving(last_velocity)

        # Branchless masked spring-buffer update: shift + append the current
        # normalized sample for the selected environments. ``torch.where`` is
        # used instead of boolean-mask indexing so the scripted forward needs no
        # implicit device synchronization.
        spring_shifted = torch.cat([self.spring_buffer[:, 1:, :], x[:, -1:, :]], dim=1)
        spring_mask3 = spring_mask.unsqueeze(1).unsqueeze(2)
        self.spring_buffer = torch.where(spring_mask3, spring_shifted, self.spring_buffer)

        # One call is one force tick: shift the force buffers for all environments.
        force_shifted = torch.cat([self.force_buffer[:, 1:, :], x[:, -1:, :]], dim=1)
        self.force_buffer.copy_(force_shifted)

        self.spring_update_counter.add_(1)

        # The spring transformer only changes the result when a spring buffer
        # updates or an env has not produced a latent yet (fresh/reset state).
        # On all-frozen ticks its output would equal the cached latent, so the
        # EMA continues from the cache and the transformer is skipped: this is
        # both faster and bit-identical to re-running the transformer on the
        # unchanged buffer.
        needs_run = spring_mask | torch.logical_not(self.anchor_valid)
        if bool(needs_run.any()):
            # Run model transformer on the normalized spring buffers to obtain latent vectors.
            latent_norm = self.model_transformer(self.spring_buffer)  # [num_envs, 1, latent_dim]
            self.latent_anchor.copy_(latent_norm)
            self.anchor_valid.fill_(True)
        else:
            # Frozen fast path: the spring buffer is identical to the last
            # transformer call, so the transformer would return the cached
            # latent again.
            latent_norm = self.latent_anchor

        # Smooth the latent estimates with exponential moving average to discourage
        # rapid switching between spring predictions.
        smoothed_latent = self.spring_alpha * latent_norm + (1.0 - self.spring_alpha) * self.last_latent
        self.last_latent.copy_(smoothed_latent)

        # Build force transformer input: [position, velocity, latent].
        # The incoming force window may be shorter than the spring buffer, so
        # expand the latent estimates to match the force transformer's input length.
        latent_channel = smoothed_latent.expand(-1, self.force_buffer.size(1), -1)
        force_input_norm = torch.cat([self.force_buffer, latent_channel], dim=-1)
        force_pred_norm = self.force_transformer(force_input_norm)  # [num_envs, 1, 1]

        # Reconstruct spring coefficients from the latent vectors.
        spring_pred_norm = self.spring_coeff_head(smoothed_latent)  # [num_envs, 1, 1]

        # Stack force and spring predictions so the wrapper can denormalize each
        # channel with its own output statistics.
        return torch.cat([force_pred_norm, spring_pred_norm], dim=-1)  # [1, 1, 2]

    def _forward_batched(self, spring_windows: torch.Tensor, force_windows: torch.Tensor) -> torch.Tensor:
        # spring_windows shape: [Batch, Spring History, Feature Dim]
        # force_windows shape: [Batch, Force History, Feature Dim]

        # Latent representation from the model transformer (input is already normalized).
        latent_norm = self.model_transformer(spring_windows)  # [Batch, 1, latent_dim]

        # The latent vector is already normalized in model-output space.
        # Repeat it across the force history dimension and feed it to the force transformer.
        latent_channel = latent_norm.expand(-1, force_windows.size(1), -1)
        force_input_norm = torch.cat([force_windows, latent_channel], dim=-1)
        force_pred_norm = self.force_transformer(force_input_norm)  # [Batch, 1, 1]

        # Reconstruct the spring coefficient from the latent vector.
        spring_pred_norm = self.spring_coeff_head(latent_norm)  # [Batch, 1, 1]

        return torch.cat([force_pred_norm, spring_pred_norm], dim=-1)  # [Batch, 1, 2]


class FrozenLatentForceModel(torch.nn.Module):
    """Deployment-only pruned force estimator for the frozen-latent mode.

    Contains strictly the components the frozen-latent path uses: the force
    transformer, the small spring-coefficient head, the stateful force buffer
    and the frozen latent. The spring transformer, spring buffer, EMA state,
    update counter and velocity gating are pruned, so the scripted model loads
    less memory and runs only the force path.

    The class is built (with the already-compiled force transformer and
    spring head of a spring transformer checkpoint) and exported by
    ``actuator_network.export_frozen_latent`` as
    ``spring_transformer_frozen_<label>.pt``. The frozen latent is embedded at
    export time (``set_frozen_latent`` can retarget it afterwards) and survives
    ``reset`` and batch-size changes, mirroring the frozen mode of
    ``SpringTransformerModel``.

    ``forward(x)`` expects ``x`` of shape ``[num_envs, 1, Feature]`` (the
    latest normalized sample only, normalized by ``ScaledModelWrapper``); one
    call is one force tick: the force buffer shifts and appends the current
    sample, the shared frozen latent is expanded across the force window and
    concatenated to the force buffer before the force transformer predicts
    ``tendon_bota_force_newton_data``. The spring-coefficient head runs on the
    frozen latent for the auxiliary channel.

    The forward pass returns a 2-channel output:
        0: ``tendon_bota_force_newton_data`` (normalized with force output stats)
        1: ``spring_coeff`` (normalized with spring output stats)
    """

    def __init__(
        self,
        force_transformer: TorchTransformerModel,
        spring_coeff_head: SpringCoefficientHead,
        latent_dim: int,
        input_size: int,
        force_history_size: int,
    ) -> None:
        super().__init__()
        self.force_transformer = force_transformer
        self.spring_coeff_head = spring_coeff_head
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.force_history_size = force_history_size

        # Stateful force buffer for online inference (one state per
        # environment; zero-initialized to match the zero-padded training
        # windows built in the normalized domain). Reassigned to the incoming
        # batch size, so the scripted model works with any num_envs.
        self.register_buffer("force_buffer", torch.zeros(1, force_history_size, input_size))
        self.register_buffer("frozen_latent", torch.zeros(1, 1, latent_dim))

    def _ensure_state(self, num_envs: int, device: torch.device) -> None:
        """Re-initialize the force buffer to zeros when the input batch size changes.

        Changing the batch size resets the force buffer (fresh session
        semantics); the frozen latent is intentionally preserved.
        """
        if self.force_buffer.size(0) != num_envs:
            self.force_buffer = torch.zeros(num_envs, self.force_history_size, self.input_size, device=device)

    @torch.jit.export
    def reset(self, reset_idx: torch.Tensor | None = None) -> None:
        """Clear the force buffer (selected environments or all).

        The frozen latent intentionally survives resets.

        Args:
            reset_idx: Optional bool tensor of length ``num_envs`` selecting the
                environments to reset. When None, all environments are reset.
        """
        if reset_idx is None:
            self.force_buffer.zero_()
            return
        mask = reset_idx.reshape(-1)
        self.force_buffer[mask] = 0.0

    @torch.jit.export
    def set_frozen_latent(self, frozen_latent: torch.Tensor) -> None:
        """(Re)set the frozen latent fed to the force transformer every tick.

        Args:
            frozen_latent: Normalized latent vector (model-output space) holding
                ``latent_dim`` elements, e.g. of shape ``[1, 1, latent_dim]``.
        """
        latent = frozen_latent.reshape(1, 1, self.latent_dim)
        self.frozen_latent.copy_(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [num_envs, 1, Feature] (already normalized by ScaledModelWrapper)
        self._ensure_state(x.shape[0], x.device)

        # One call is one force tick: shift the force buffers for all environments.
        force_shifted = torch.cat([self.force_buffer[:, 1:, :], x[:, -1:, :]], dim=1)
        self.force_buffer.copy_(force_shifted)

        # The frozen latent is shared across environments: expand it across the
        # force history dimension and feed it to the force transformer.
        frozen_latent = self.frozen_latent.expand(x.size(0), -1, -1)
        latent_channel = frozen_latent.expand(-1, self.force_buffer.size(1), -1)
        force_input_norm = torch.cat([self.force_buffer, latent_channel], dim=-1)
        force_pred_norm = self.force_transformer(force_input_norm)  # [num_envs, 1, 1]

        # Reconstruct the spring coefficient from the frozen latent.
        spring_pred_norm = self.spring_coeff_head(frozen_latent)  # [num_envs, 1, 1]

        return torch.cat([force_pred_norm, spring_pred_norm], dim=-1)  # [num_envs, 1, 2]


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
