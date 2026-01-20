import torch

from helpers.wrapper import ScaledModelWrapper


def split_data(inputs, outputs, train_ratio=0.8):
    """Split inputs and outputs into training and validation sets
    Args:
        inputs (torch.Tensor): Input tensor of shape (num_samples, input_dim)
        outputs (torch.Tensor): Output tensor of shape (num_samples, output_dim)
        train_ratio (float): Ratio of data to use for training
    Returns:
        tuple: (train_inputs, train_outputs, val_inputs, val_outputs)
    """
    num_samples = inputs.shape[0]
    train_size = int(num_samples * train_ratio)

    train_inputs = inputs[:train_size]
    train_outputs = outputs[:train_size]
    val_inputs = inputs[train_size:]
    val_outputs = outputs[train_size:]

    return train_inputs, train_outputs, val_inputs, val_outputs


def data_generator(inputs, outputs, batch_size):
    """Generate batches of data
    Args:
        inputs (torch.Tensor): Input tensor of shape (num_samples, input_dim)
        outputs (torch.Tensor): Output tensor of shape (num_samples, output_dim)
        batch_size (int): Size of each batch
    Yields:
        tuple: (batch_inputs, batch_outputs)
    """
    num_samples = inputs.shape[0]
    indices = torch.randperm(num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield inputs[batch_indices], outputs[batch_indices]


def train(model, inputs, outputs, model_to_save: ScaledModelWrapper = None):
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 64

    inputs_train, outputs_train, inputs_val, outputs_val = split_data(inputs, outputs)
    data_generator_train = data_generator(inputs_train, outputs_train, batch_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for batch_inputs, batch_outputs in data_generator_train:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                test_loss = criterion(model(inputs_val), outputs_val)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {test_loss.item():.4f}")

    if model_to_save is not None:
        model_to_save.freeze()
        model_to_save.trace_and_save("model_scripted.pt")
        model_to_save.unfreeze()
        torch.save(model_to_save.state_dict(), "model.pth")
