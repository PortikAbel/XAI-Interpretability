import torch.nn as nn
from abc import abstractmethod

class AbstractModel(nn.Module):
    def __init__(self, model):
        """
        An abstract wrapper for PyTorch models implementing functions required for evaluation.
        Args:
            model: PyTorch neural network model
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, input):
        return self.model