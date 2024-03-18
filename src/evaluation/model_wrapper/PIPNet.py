from evaluation.model_wrapper.base import AbstractModel


class PipNetModel(AbstractModel):
    """
    A wrapper for PipPNet models.
    Args:
        model: PyTorch ProtoPNet model
    """

    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def forward(self, input, return_partial_outputs=False):
        proto_features, pooled, out = self.model(input)
        if not return_partial_outputs:
            return out
        else:
            return proto_features, pooled, out

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
