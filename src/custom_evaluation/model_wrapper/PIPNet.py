from custom_evaluation.model_wrapper.base import AbstractModel

class PipNetModel(AbstractModel):
    """
    A wrapper for PipPNet models.
    Args:
        model: PyTorch ProtoPNet model
    """
    def __init__(self, model, load_model_dir, epoch_number_str):
        super().__init__(model)
        self.model = model
        self.load_model_dir = load_model_dir
        self.epoch_number_str = epoch_number_str

    def forward(self, input, return_pooled = False):
        proto_features, pooled, out = self.model(input)
        if not return_pooled:
            return out
        else:
            return pooled, out

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)