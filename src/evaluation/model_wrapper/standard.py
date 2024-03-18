from evaluation.model_wrapper.base import AbstractModel


class StandardModel(AbstractModel):
    """
    A wrapper for standard PyTorch models (e.g. ResNet, VGG, AlexNet, ...).
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        return self.model(input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
