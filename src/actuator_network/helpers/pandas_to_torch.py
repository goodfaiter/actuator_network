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
