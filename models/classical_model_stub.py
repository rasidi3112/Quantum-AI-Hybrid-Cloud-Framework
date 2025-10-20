# Dummy stub untuk pytest
from .classical_model import ClassicalModelConfig

class ClassicalFeatureExtractor:
    def __init__(self, config: ClassicalModelConfig):
        self.config = config

    def forward(self, x):
        # kembalikan tensor dummy dengan dimensi output sesuai config
        import torch # type: ignore
        batch_size = x.shape[0]
        output_dim = self.config.output_dim
        return torch.zeros(batch_size, output_dim)
