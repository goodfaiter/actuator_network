import torch
import math


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

    def deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchRNNModel(torch.nn.Module):
    """RNN Model with PyTorch"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, device: torch.device):
        super(TorchRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True, device=device)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=output_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1)  # Unsqueeze to keep consistent output shape

    def deploy_forward(self, x: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out, hn


class TorchTransformerModel(torch.nn.Module):
    def __init__(
        self, input_size: int, output_size: int, num_layers: int, seq_length: int, num_heads: int, hidden_dim: int, device: torch.device
    ):
        super(TorchTransformerModel, self).__init__()

        # Input projection
        self.input_projection = torch.nn.Linear(input_size, hidden_dim, device=device)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(max_len=seq_length, hidden_dim=hidden_dim, device=device)

        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            device=device,
            dropout=0.1,
            activation=torch.nn.Tanh(),
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output layer (taking only the last timestep)
        self.output_sequence = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim // 4, output_size, device=device),
        )

        # Causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length).to(device)
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

        # Take only the last timestep and output
        x = x[:, -1, :]  # Take last timestep
        output = self.output_sequence(x)

        return output.unsqueeze(1)  # Unsqueeze to keep consistent output shape

    def deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


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
