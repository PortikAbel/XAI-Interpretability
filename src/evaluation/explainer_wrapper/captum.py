from evaluation.explainer_wrapper.base import AbstractAttributionExplainer


class CaptumAttributionExplainer(AbstractAttributionExplainer):
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """

    def explain(self, input, target=None):
        if self.explainer_name == "InputXGradient":
            return self.explainer.attribute(input, target=target)
        elif self.explainer_name == "IntegratedGradients":
            return self.explainer.attribute(
                input, target=target, baselines=self.baseline, n_steps=50
            )
