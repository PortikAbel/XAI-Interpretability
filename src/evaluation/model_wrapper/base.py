from abc import abstractmethod

import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(self, model):
        """
        An abstract wrapper for PyTorch models implementing
        functions required for evaluation.

        Args:
            model: PyTorch neural network model
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, input_):
        return self.model

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()
