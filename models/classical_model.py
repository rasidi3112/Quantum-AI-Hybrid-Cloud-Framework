from torch import nn # type: ignore

class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        input_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if config.activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif config.activation.lower() == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClassicalModelConfig:
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], activation: str, dropout: float):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
