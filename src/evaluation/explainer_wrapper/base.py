from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn


class AbstractExplainer:
    def __init__(self, explainer, baseline=None):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.explainer_name = type(self.explainer).__name__
        self.baseline = baseline
        print(self.explainer_name)

    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)


class AbstractAttributionExplainer(AbstractExplainer):
    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)

    def get_important_parts(
        self, image, part_map, target, colors_to_part, thresholds, with_bg=False
    ):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1  # B = 1
        attribution = self.explain(image, target=target)

        part_importances = self.get_part_importance(
            image, part_map, target, colors_to_part, with_bg=with_bg
        )

        important_parts_for_thresholds = []

        for threshold in thresholds:
            important_parts = []
            for key in part_importances.keys():
                if part_importances[key] > (attribution.sum() * threshold):
                    important_parts.append(key)
            important_parts_for_thresholds.append(important_parts)
        return important_parts_for_thresholds

    def get_part_importance(
        self, image, part_map, target, colors_to_part, with_bg=False
    ):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: {'beak': 0.5, 'wing':, 'tail':}
        """
        assert image.shape[0] == 1  # B = 1
        attribution = self.explain(image, target=target)

        part_importances = {}

        dilation1 = nn.MaxPool2d(5, stride=1, padding=2)
        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1, 3, 1, 1).to(image.device)
            torch_color[0, 0, 0, 0] = part_color[0]
            torch_color[0, 1, 0, 0] = part_color[1]
            torch_color[0, 2, 0, 0] = part_color[2]
            color_available = torch.all(
                part_map == torch_color, dim=1, keepdim=True
            ).float()

            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            attribution_in_part = attribution_in_part.sum()

            part_string = colors_to_part[part_color]
            part_string = "".join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()

        if with_bg:
            for i in range(50):  # TODO: adjust 50 if more background parts are used
                torch_color = torch.zeros(1, 3, 1, 1).to(image.device)
                torch_color[0, 0, 0, 0] = 204
                torch_color[0, 1, 0, 0] = 204
                torch_color[0, 2, 0, 0] = 204 + i
                color_available = torch.all(
                    part_map == torch_color, dim=1, keepdim=True
                ).float()
                color_available_dilated = dilation1(color_available)

                attribution_in_part = attribution * color_available_dilated
                attribution_in_part = attribution_in_part.sum()

                bg_string = "bg_" + str(i).zfill(3)
                part_importances[bg_string] = attribution_in_part.item()

        return part_importances

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)
