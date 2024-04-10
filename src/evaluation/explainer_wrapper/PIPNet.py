import cv2
import torch
import torch.nn as nn

from evaluation.explainer_wrapper.base import AbstractAttributionExplainer
from models.ProtoPNet.util.helpers import find_high_activation_crop


# this should in the end be the final explainer
# this explainer can also be used for the visualizations to clean up the code a bit
class PIPNetExplainer(AbstractAttributionExplainer):
    """
    A wrapper for PIPNet.
    Args:
        model: PyTorch model.
    """

    def __init__(self, model, explainer):
        """
        A wrapper for PIPNet explanations.
        Args:
            model: PyTorch neural network model
        """
        super().__init__(explainer)
        self.model = model
        self.dilation = nn.MaxPool2d(1, stride=1, padding=0)

    find_high_activation_crop = find_high_activation_crop

    # for evaluating pipnet explainations are masks
    def explain(self, image, target):
        B, C, H, W = image.shape
        idx = 0

        proto_features, pooled, _out = self.model(image, return_partial_outputs=True)
        target_class = target[idx]

        sim_weights = (
            pooled[idx, :]
            * self.model.model.module._classification.weight[target_class, :]
        )
        _, sorted_proto_indices = torch.sort(sim_weights)
        sorted_proto_indices = sorted_proto_indices[
            sim_weights[sorted_proto_indices] > 0.1
        ]

        self.inference_image_masks = []
        self.similarity_scores = []
        self.class_connections = []
        self.bounding_box_coords = []
        self.prototype_idxs = []

        attribution = torch.zeros_like(image)

        for prototype_index in reversed(sorted_proto_indices.detach().cpu().numpy()):
            self.prototype_idxs.append(prototype_index)

            activation_pattern = (
                proto_features[idx][prototype_index].detach().cpu().numpy()
            )
            upsampled_activation_pattern = cv2.resize(
                activation_pattern, dsize=(H, W), interpolation=cv2.INTER_CUBIC
            )
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = self.find_high_activation_crop(
                upsampled_activation_pattern
            )

            mask = torch.zeros_like(image)
            mask[
                :,
                :,
                high_act_patch_indices[0] : high_act_patch_indices[1],
                high_act_patch_indices[2] : high_act_patch_indices[3],
            ] = 1

            inference_image_mask = mask.to(image.device)
            similarity_score = pooled[idx][prototype_index]
            class_connection = self.model.model.module._classification.weight[
                target_class
            ][prototype_index]
            attribution += inference_image_mask * similarity_score * class_connection

            self.inference_image_masks.append(mask)
            self.similarity_scores.append(similarity_score)
            self.class_connections.append(class_connection)
            self.bounding_box_coords.append(high_act_patch_indices)

        return attribution

    def get_important_parts(
        self, image, part_map, target, colors_to_part, thresholds, with_bg=False
    ):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1  # B = 1
        self.explain(image, target=target)
        attribution = torch.zeros_like(image)
        for inference_image_mask in self.inference_image_masks:
            inference_image_mask = inference_image_mask.to(image.device)
            attribution = attribution + inference_image_mask
        attribution = attribution.clamp(min=0.0, max=1.0)

        colors_to_part = {
            k: "".join(filter(str.isalpha, v)) for k, v in colors_to_part.items()
        }
        if with_bg:
            for i in range(50):  # TODO: adjust 50 if more background parts are used
                colors_to_part[(204, 204, 204 + i)] = f"bg_{str(i).zfill(3)}"

        important_parts_for_thresholds = []

        for threshold in thresholds:
            important_parts = set()
            for color, part_name in colors_to_part.items():
                torch_color = torch.tensor(color).to(image.device)[None, :, None, None]
                part_mask = torch.all(part_map == torch_color, dim=1, keepdim=True)
                attribution_in_part = attribution * part_mask

                # threshold to decide how big attribution in part should be
                if attribution_in_part.sum() > threshold * part_mask.sum():
                    important_parts.add(part_name)

            important_parts_for_thresholds.append(list(important_parts))

        return important_parts_for_thresholds
