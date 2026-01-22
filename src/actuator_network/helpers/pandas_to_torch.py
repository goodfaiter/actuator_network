import torch


def normalize_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Normalize a tensor to have zero mean and unit variance
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, feature_dim)
    Returns:
        tuple: Normalized tensor, mean, and standard deviation
    """
    mean = torch.mean(tensor, dim=0, keepdim=True)
    std = torch.std(tensor, dim=0, keepdim=True) + 1e-8  # Add small value to avoid division by zero
    normalized_tensor = (tensor - mean) / std

    return normalized_tensor, mean, std


def process_inputs(data: torch.Tensor, stride: int, num_hist: int, prediction: bool) -> torch.Tensor:
    """ Create history vectors
    Args:
        data (torch.Tensor): Input tensor of shape (batch_size, feature_dim)
        stride (int): Stride between history steps
        num_hist (int): Number of history steps to include
        prediction (bool): Whether this is for prediction or estimation (affects offset)
    Returns:
        torch.Tensor: Tensor with history vectors of shape (batch_size, feature_dim * num_hist)
    """
    batch_size, feature_dim = data.shape
    history_vector = []
    for i in range(batch_size):
        if i + (num_hist + (0 if prediction else -1)) * stride >= batch_size:
            break
        one_step = torch.zeros((num_hist * feature_dim), device=data.device)
        for j in range(num_hist):
            one_step[j * feature_dim:(j + 1) * feature_dim] = data[i + j * stride]
        history_vector.append(one_step)

    hist_tensor = torch.stack(history_vector)

    return hist_tensor.to(data.device)

def process_outputs(data: torch.Tensor, stride: int, num_hist: int, prediction: bool) -> torch.Tensor:
    """ Create future output vectors
    Args:
        data (torch.Tensor): Input tensor of shape (batch_size, feature_dim)
        stride (int): Stride between future steps
        num_hist (int): Number of history steps
        prediction (bool): Whether this is for prediction or estimation (affects offset)
    Returns:
        torch.Tensor: Tensor with future output vectors of shape (batch_size, feature_dim)
    """
    batch_size, feature_dim = data.shape
    history_vector = []
    for i in range(batch_size):
        if i + (num_hist + (0 if prediction else -1)) * stride >= batch_size:
            break
        one_step = torch.zeros((feature_dim), device=data.device)
        one_step[:] = data[i + (num_hist + (0 if prediction else -1)) * stride] # + since we're predicting the next step after history
        history_vector.append(one_step)
    
    hist_tensor = torch.stack(history_vector)

    return hist_tensor.to(data.device)

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
