from pyparsing import Optional
import torch


class TorchMlpModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: list):
        super(TorchMlpModel, self).__init__()
        layers = []
        in_size = input_size

        for hidden_size in hidden_layers:
            layers.append(torch.nn.Linear(in_size, hidden_size))
            layers.append(torch.nn.ReLU())
            in_size = hidden_size

        layers.append(torch.nn.Linear(in_size, output_size))
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

        self.register_buffer('_h0', torch.zeros(num_layers, 1, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x, self._h0.expand(self.num_layers, x.size(0), self.hidden_size))
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1) # Unsqueeze to keep consistent output shape
    
    def deploy_forward(self, x: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        
        return out, hn