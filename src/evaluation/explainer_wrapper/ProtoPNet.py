import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from evaluation.explainer_wrapper.base import AbstractAttributionExplainer


# this should in the end be the final explainer
# this explainer can also be used for the visualizations to clean up the code a bit
class ProtoPNetExplainer(AbstractAttributionExplainer):
    """
    A wrapper for ProtoPNet.
    Args:
        model: PyTorch model.
    """

    def __init__(self, model):
        """
        A wrapper for ProtoPNet explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model
        self.load_model_dir = model.load_model_dir
        if self.load_model_dir.is_file():
            self.load_model_dir = self.load_model_dir.parent
        self.load_model_dir = self.load_model_dir.parent
        self.epoch_number = model.epoch_number
        self.dilation = nn.MaxPool2d(1, stride=1, padding=0)

    def find_high_activation_crop(self, activation_map, percentile=95):
        threshold = np.percentile(activation_map, percentile)
        mask = np.ones(activation_map.shape)
        mask[activation_map < threshold] = 0
        lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
        for i in range(mask.shape[0]):
            if np.amax(mask[i]) > 0.5:
                lower_y = i
                break
        for i in reversed(range(mask.shape[0])):
            if np.amax(mask[i]) > 0.5:
                upper_y = i
                break
        for j in range(mask.shape[1]):
            if np.amax(mask[:, j]) > 0.5:
                lower_x = j
                break
        for j in reversed(range(mask.shape[1])):
            if np.amax(mask[:, j]) > 0.5:
                upper_x = j
                break
        return lower_y, upper_y, lower_x, upper_x

    # for evaluating protopnet explainations are masks
    def explain(self, image, target):
        B, C, H, W = image.shape

        idx = 0

        logits, additional_outs = self.model(image, return_min_distances=True)
        model = self.model.model
        if type(model) is torch.nn.DataParallel:
            model = model.module
        conv_output, distances = model.push_forward(image)
        prototype_activations = model.distance_2_similarity(
            additional_outs.min_distances
        )
        prototype_activation_patterns = model.distance_2_similarity(distances)

        target_class = target[idx]

        class_prototype_indices = np.nonzero(
            model.prototype_class_identity.detach().cpu().numpy()[:, target_class]
        )[0]
        class_prototype_activations = prototype_activations[idx][
            class_prototype_indices
        ]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        self.inference_image_masks = []
        self.prototypes = []  # these are the training set prototypes
        self.prototype_idxs = []  # these are the training set prototypes
        self.similarity_scores = []
        self.class_connections = []
        self.bounding_box_coords = []

        attribution = torch.zeros_like(image)

        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]
            self.prototype_idxs.append(prototype_index)

            prototype = plt.imread(
                self.load_model_dir
                / "visualization_results"
                / f"epoch-{self.epoch_number}"
                / f"prototype-img{prototype_index.item()}.png"
            )
            prototype = cv2.cvtColor(np.uint8(255 * prototype), cv2.COLOR_RGB2BGR)
            prototype = prototype[..., ::-1]
            self.prototypes.append(prototype)

            activation_pattern = (
                prototype_activation_patterns[idx][prototype_index]
                .detach()
                .cpu()
                .numpy()
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
            similarity_score = prototype_activations[idx][prototype_index]
            class_connection = model.last_layer.weight[target_class][prototype_index]
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
