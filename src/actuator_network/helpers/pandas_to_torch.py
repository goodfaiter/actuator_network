import torch


def normalize_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize a tensor to have zero mean and unit variance
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, ..., feature_dim)
    Returns:
        tuple: Normalized tensor, mean, and standard deviation
    """
    mean = torch.zeros(1, tensor.shape[-1], device=tensor.device, requires_grad=False)
    std = torch.ones(1, tensor.shape[-1], device=tensor.device, requires_grad=False)
    mean[:] = torch.mean(
        tensor,
        dim=[i for i in range(tensor.dim() - 1)],
        keepdim=True,
    )
    std[:] = (
        torch.std(tensor, dim=[i for i in range(tensor.dim() - 1)], keepdim=True) + 1e-8
    )  # Add small value to avoid division by zero
    with torch.no_grad():
        normalized_tensor = (tensor - mean) / std

    return normalized_tensor, mean, std


def apply_normalization(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply pre-computed mean and standard deviation to normalize a tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, ..., feature_dim).
        mean (torch.Tensor): Mean tensor of shape (1, feature_dim).
        std (torch.Tensor): Standard deviation tensor of shape (1, feature_dim).

    Returns:
        torch.Tensor: Normalized tensor with the same shape as ``tensor``.
    """
    with torch.no_grad():
        return (tensor - mean) / std


def process_inputs_time_series(data: torch.Tensor, history_size: int, stride: int) -> torch.Tensor:
    """Turn inputs into short sequences of time series looking forward.

    Args:
        data (torch.Tensor): Input tensor of shape (batch_size, feature_dim)
        history_size (int): Length of the input sequences
        stride (int): Stride between history steps

    Returns:
        torch.Tensor: Tensor with input sequences of shape (batch_size, history_size, feature_dim)
    """
    batch_size, feature_dim = data.shape
    num_sequences = batch_size - (history_size - 1) * stride
    if num_sequences <= 0:
        return torch.empty((0, history_size, feature_dim), device=data.device)

    offsets = torch.arange(history_size, device=data.device) * stride
    indices = torch.arange(num_sequences, device=data.device).unsqueeze(1) + offsets.unsqueeze(0)

    return data[indices]


def process_outputs_time_series(data: torch.Tensor, stride: int, history_size: int) -> torch.Tensor:
    """Create output vectors matching the last index of each input sequence.

    Args:
        data (torch.Tensor): Input tensor of shape (batch_size, feature_dim)
        stride (int): Stride between history steps
        history_size (int): Number of history steps in each input sequence

    Returns:
        torch.Tensor: Tensor with output vectors of shape (batch_size, 1, feature_dim)
    """
    batch_size, feature_dim = data.shape
    num_sequences = batch_size - (history_size - 1) * stride
    if num_sequences <= 0:
        return torch.empty((0, 1, feature_dim), device=data.device)

    target_index = torch.arange(num_sequences, device=data.device) + (history_size - 1) * stride
    return data[target_index].unsqueeze(1)


def pandas_to_torch(df, device="cpu"):
    """
    Convert a pandas DataFrame to a PyTorch tensor.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to convert.
    device (str): The device to load the tensor onto ('cpu' or 'cuda').

    Returns:
    torch.Tensor: The resulting PyTorch tensor.
    """
    np_array = df.to_numpy()
    col_indices = df.columns.tolist()
    tensor = torch.tensor(np_array, dtype=torch.float32, device=device)

    return col_indices, tensor


def build_strided_windows(data: torch.Tensor, history_size: int, stride: int) -> torch.Tensor:
    """Build zero-padded sliding windows sampled backward at the given stride.

    Each window ends at its own raw index and samples backward with the given
    stride between history samples. Early timesteps for which the history
    would extend before the start of the data are included and padded with
    zeros at the beginning of the window, so one window is produced per sample.

    Args:
        data: Input tensor of shape ``(batch_size, feature_dim)``.
        history_size: Length of each input window.
        stride: Stride between history samples inside a window.

    Returns:
        Tensor of shape ``(batch_size, history_size, feature_dim)``.
    """
    batch_size, feature_dim = data.shape
    if batch_size == 0:
        return torch.empty((0, history_size, feature_dim), device=data.device)

    end_indices = torch.arange(batch_size, device=data.device)
    offsets = torch.arange(history_size, device=data.device) * stride - (history_size - 1) * stride
    indices = end_indices.unsqueeze(1) + offsets.unsqueeze(0)
    indices_clamped = indices.clamp_min(0)
    windows = data[indices_clamped].clone()
    windows[indices < 0] = 0.0
    return windows


def build_aligned_windows(
    data: torch.Tensor,
    spring_history_size: int,
    force_history_size: int,
    spring_stride: int,
    force_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build zero-padded spring and force windows aligned to the same end timestep.

    Both windows end at the same raw index, but each window samples backward at
    its own stride. This is necessary because the two transformers may use
    different history lengths and different strides.

    Early timesteps for which the history would extend before the start of the
    data are included and padded with zeros at the beginning of the window.

    Args:
        data: Input tensor of shape ``(batch_size, feature_dim)``.
        spring_history_size: Length of the spring transformer's input window.
        force_history_size: Length of the force transformer's input window.
        spring_stride: Stride between spring history samples.
        force_stride: Stride between force history samples.

    Returns:
        Tuple of ``(spring_windows, force_windows)`` with shapes
        ``(batch_size, spring_history_size, feature_dim)`` and
        ``(batch_size, force_history_size, feature_dim)``.
    """
    spring_windows = build_strided_windows(data, spring_history_size, spring_stride)
    force_windows = build_strided_windows(data, force_history_size, force_stride)
    return spring_windows, force_windows


def build_frozen_spring_windows(
    normal_windows: torch.Tensor,
    velocity_idx: int,
    threshold_lo: float,
    threshold_hi: float,
) -> torch.Tensor:
    """Build spring windows where the buffer is frozen while the velocity stays within bounds.

    Args:
        normal_windows: Normalized sliding windows of shape [N, H, F].
        velocity_idx: Index of the velocity channel.
        threshold_lo: Normalized lower threshold bound; the buffer updates when the
            velocity falls below it.
        threshold_hi: Normalized upper threshold bound; the buffer updates when the
            velocity rises above it.

    Returns:
        Spring windows of the same shape as ``normal_windows``.
    """
    num_samples = normal_windows.size(0)
    spring_windows = normal_windows.clone()
    last_moving_window = torch.zeros_like(normal_windows[0])

    for i in range(num_samples):
        velocity = normal_windows[i, -1, velocity_idx]
        if (velocity > threshold_hi) | (velocity < threshold_lo):
            last_moving_window = normal_windows[i].clone()
        spring_windows[i] = last_moving_window

    return spring_windows
