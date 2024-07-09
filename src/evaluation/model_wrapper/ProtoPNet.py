from evaluation.model_wrapper.base import AbstractModel


class ProtoPNetModel(AbstractModel):
    """
    A wrapper for ProtoPNet models.
    Args:
        model: PyTorch ProtoPNet model
    """

    def __init__(self, model, load_model_dir, epoch_number):
        super().__init__(model)
        self.model = model
        self.load_model_dir = load_model_dir
        self.epoch_number = epoch_number

    def forward(self, input_, return_min_distances=False):
        logits, min_distances = self.model(input_)
        if not return_min_distances:
            return logits
        else:
            return logits, min_distances
